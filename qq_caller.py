import asyncio
import base64
import json
import time
import os
import traceback
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

try:
    import torch
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )
    _silero_model.eval()
    _silero_get_speech_ts = _silero_utils[0]
    print("✅ Silero VAD model loaded")
except Exception as _silero_load_err:
    _silero_model = None
    _silero_get_speech_ts = None
    print(f"⚠️ Silero VAD failed to load: {_silero_load_err} — barge-in VAD disabled")

import plivo

from fastapi import WebSocket
from google import genai
from google.genai import types

from iae_knowledge import IEI_KNOWLEDGE
from system_prompticallend import build_system_message, TOOL_DECLARATIONS
from transcript_service import (
    add_conversation_entry, update_call_end,
    generate_auto_summary, get_ist_string
)
from appointment_service import store_caller_info, book_appointment
from caller_memory import load_caller_memory, save_caller_memory, save_call_summary, build_memory_context

DEBUG = os.getenv("DEBUG_TOOL_CALLS", "1") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
MAX_SESSION_RETRIES = 3

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

PUBLIC_URL = os.getenv("PUBLIC_URL")
HUMAN_AGENT_NUMBER = os.getenv("HUMAN_AGENT_NUMBER")

from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("MONGO_URI"))

_plivo_client = None
def get_plivo_client():
    global _plivo_client
    if _plivo_client is None:
        _plivo_client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
    return _plivo_client


async def hangup_call(call_uuid: str):
    try:
        client = get_plivo_client()
        await asyncio.to_thread(client.calls.hangup, call_uuid)
        print(f"☎️ [END CALL] Hung up call_uuid={call_uuid}")
    except Exception as e:
        print(f"[HANGUP ERROR] type={type(e).__name__} msg={e}")


async def transfer_call(call_uuid: str, transfer_xml_url: str):
    try:
        client = get_plivo_client()
        await asyncio.to_thread(
            client.calls.transfer,
            call_uuid,
            legs="aleg",
            aleg_url=transfer_xml_url
        )
        print(f"☎️ [TRANSFER CALL] Transferring call_uuid={call_uuid} to {transfer_xml_url}")
    except Exception as e:
        print(f"[TRANSFER ERROR] type={type(e).__name__} msg={e}")

# ── Silero VAD constants ──
SILERO_THRESHOLD = 0.25
FRAMES_REQUIRED = 3
INTERRUPT_COOLDOWN = 800
INTERRUPT_HOLD_MS = 300
MIN_SPEECH_STOP_GAP_MS = 200

# ── Slow tools (kept for reference, hold audio trigger removed) ──
SLOW_TOOLS = {"fetch_url_content", "query_iei_knowledge_rag", "search_iei_knowledge"}

active_calls = {}
stream_call_map = {}

_gemini_client = None

def _load_hold_audio_from_mp3(path: str = "hold.mp3") -> str:
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(path)
        audio = audio.set_channels(1).set_frame_rate(8000).set_sample_width(2)
        pcm8k = audio.raw_data
        mulaw = audioop.lin2ulaw(pcm8k, 2)
        b64 = base64.b64encode(mulaw).decode()
        print(f"✅ Hold audio loaded from {path} ({len(mulaw)} bytes mulaw)")
        return b64
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e} — hold audio disabled")
        return ""

_hold_audio_b64_global = _load_hold_audio_from_mp3("hold.mp3")
_transfer_audio_b64_global = _load_hold_audio_from_mp3("call center.mp3")


def get_gemini_client(api_key):
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def normalize_phone(value):
    if not isinstance(value, str):
        return ""
    cleaned = ''.join(c for c in value if c.isdigit() or c == '+')
    return cleaned or value.strip()


def resolve_caller_phone(data, fallback="unknown"):
    start = data.get("start", {})
    candidate = (
        start.get("from") or
        start.get("caller") or
        data.get("from") or
        data.get("caller") or
        fallback
    )
    return normalize_phone(candidate) or fallback


def dump_response_safe(response):
    try:
        if hasattr(response, "model_dump"):
            d = response.model_dump(exclude_none=True)
            return json.dumps(d, default=str)[:2000]
    except Exception as dump_err:
        return f"<dump failed: {type(dump_err).__name__}: {dump_err}> repr={repr(response)[:500]}"
    return repr(response)[:500]


def plivo_to_gemini(payload: str) -> bytes:
    mulaw = base64.b64decode(payload)
    pcm8k = audioop.ulaw2lin(mulaw, 2)
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return pcm16k


def gemini_to_plivo(pcm24k: bytes) -> str:
    pcm8k, _ = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, None)
    pcm8k = audioop.mul(pcm8k, 2, 3.0)
    mulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(mulaw).decode()


async def is_speech_silero(pcm16k: bytes) -> bool:
    if _silero_model is None:
        return False
    try:
        frame = pcm16k[:1024]
        if len(frame) < 1024:
            frame = frame + b'\x00' * (1024 - len(frame))

        def _run_silero():
            import torch
            audio_tensor = torch.frombuffer(frame, dtype=torch.int16).float() / 32768.0
            with torch.no_grad():
                confidence = _silero_model(audio_tensor, 16000).item()
            return confidence >= SILERO_THRESHOLD

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_silero)
    except Exception:
        return False


def build_function_declarations():
    return [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=types.Schema(
                type=t["parameters"]["type"],
                properties={
                    k: types.Schema(
                        type=v["type"],
                        description=v.get("description", ""),
                        enum=v.get("enum")
                    )
                    for k, v in t["parameters"]["properties"].items()
                },
                required=t["parameters"].get("required", [])
            )
        )
        for t in TOOL_DECLARATIONS
    ]

FUNCTION_DECLARATIONS = build_function_declarations()


def build_gemini_config(is_inbound=True, campaign_data=None, lead_data=None, resumption_handle=None, memory_context=""):
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        temperature=0.4,
        thinking_config=types.ThinkingConfig(
            thinking_budget=512
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=200,
                silence_duration_ms=600
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(
                text=build_system_message(is_inbound, campaign_data, lead_data, memory_context)
            )]
        ),
        tools=[types.Tool(function_declarations=FUNCTION_DECLARATIONS)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        session_resumption=types.SessionResumptionConfig(
            handle=resumption_handle
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    )


async def execute_tool(name, args, caller_phone, request_uuid, appointments_col, inbound_lead_col, db=None):
    if name == "query_company_data":
        topic = args.get("topic", "")
        if DEBUG:
            print(f"[TOOL] query_company_data topic={topic}")
        return IEI_KNOWLEDGE if topic == "iei" else "No data available."

    elif name == "store_caller_info":
        if DEBUG:
            print(f"[TOOL] store_caller_info args={args}")
        result = await store_caller_info(
            name=args.get("name"),
            company=args.get("company", "Not Provided"),
            phone_number=caller_phone,
            request_uuid=request_uuid,
            inbound_lead_collection=inbound_lead_col
        )
        return result

    elif name == "book_appointment":
        if DEBUG:
            print(f"[TOOL] book_appointment args={args}")
        result = await book_appointment(
            name=args.get("name"),
            date=args.get("date"),
            purpose=args.get("purpose"),
            phone_number=caller_phone,
            request_uuid=request_uuid,
            appointments_collection=appointments_col
        )
        if db is not None and caller_phone != "unknown" and "successfully" in str(result).lower():
            try:
                await save_call_summary(
                    caller_phone,
                    f"Booked appointment for {args.get('date')} regarding {args.get('purpose')}",
                    db
                )
            except Exception as e:
                print(f"[MEMORY SAVE ERROR - book_appointment] {e}")
        return result

    elif name == "fetch_url_content":
        url = args.get("url", "")
        if DEBUG:
            print(f"[TOOL] fetch_url_content url={url}")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, follow_redirects=True)
                return resp.text[:3000]
        except Exception as e:
            return f"Failed to fetch URL: {e}"

    elif name == "search_iei_knowledge":
        query = args.get("query", "")
        top_k = int(args.get("top_k", 3))
        if DEBUG:
            print(f"[TOOL] search_iei_knowledge query={query!r} top_k={top_k}")
        try:
            from pymongo import MongoClient
            from google import genai as _genai

            embed_client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            embed_result = embed_client.models.embed_content(
                model="gemini-embedding-001",
                contents=query
            )
            query_vector = embed_result.embeddings[0].values

            mongo_client = MongoClient(os.getenv("MONGO_URI"))
            col = mongo_client[os.getenv("MONGO_DB", "iei_voice_agent")]["iei_knowledge_chunks"]

            results = list(col.aggregate([
                {
                    "$vectorSearch": {
                        "index": "iei_vector_idx",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "heading": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "_id": 0
                    }
                }
            ]))
            mongo_client.close()

            if not results:
                return "No relevant information found in the IEI knowledge base."

            return "\n\n---\n\n".join(
                f"[{r['heading']}]\n{r['content']}" for r in results
            )
        except Exception as e:
            print(f"[RAG ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            return f"RAG search failed: {e}"

    return "Tool not found."


async def background_tool(fn_id, fn_name, fn_args, caller_phone, request_uuid,
                           appointments_col, inbound_lead_col,
                           session, add_function_entry_fn, pending_calls, db=None):
    try:
        result = await execute_tool(
            fn_name, fn_args,
            caller_phone, request_uuid,
            appointments_col, inbound_lead_col,
            db
        )
        await add_function_entry_fn(fn_name, fn_args, result)
        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=fn_id,
                    name=fn_name,
                    response={"result": result},
                    scheduling="WHEN_IDLE"
                )
            ]
        )
    except Exception as e:
        print(f"[BG TOOL ERROR] name={fn_name} type={type(e).__name__} msg={e}")
        traceback.print_exc()
    finally:
        pending_calls.discard(fn_id)


async def handle_media_stream(
    plivo_ws: WebSocket,
    db,
    inbound_col,
    outbound_col,
    appointments_col,
    inbound_lead_col,
    broadcast_fn=None,
    gemini_api_key=None,
):
    await plivo_ws.accept()
    print("✅ Plivo connected")
    print(f"[MODEL] Using GEMINI_MODEL = {GEMINI_MODEL}")

    request_uuid = None
    caller_phone = "unknown"
    is_inbound = True
    call_ended = False
    stream_id = None
    last_ai_entry = ""
    last_ai_entry_at = 0
    call_info = None
    campaign_data = None
    lead_data = None
    audio_buffer = []
    resumption_handle = None
    plivo_closed = False
    is_transferring = False

    # ── Caller memory state ──
    memory_context = ""
    last_known_name = ""
    last_known_company = ""
    caller_info_saved = False

    # ── Silero VAD state ──
    speech_frame_count = 0
    last_interrupt_at = 0
    speech_active = False
    speech_active_since = 0
    last_silence_at = 0
    speech_start_ms = 0

    # ── Echo gate state ──
    ai_last_spoke_at = 0

    if _silero_model is not None:
        print(f"✅ Silero VAD ready (threshold={SILERO_THRESHOLD})")
    else:
        print("⚠️ Silero VAD not available")

    _hold_audio_b64 = _hold_audio_b64_global

    async def add_entry(entry):
        await add_conversation_entry(
            request_uuid, entry, is_inbound,
            db, inbound_col, outbound_col, broadcast_fn
        )

    async def add_user_entry(text):
        text = str(text or "").strip()
        if not text:
            return
        print(f"[{get_ist_string()}] 👤 USER: {text}")
        await add_entry({
            "type": "user",
            "text": text,
            "timestamp": int(time.time() * 1000),
            "istTime": get_ist_string()
        })

    async def add_ai_entry(text):
        nonlocal last_ai_entry, last_ai_entry_at
        text = str(text or "").strip()
        if not text:
            return
        now = int(time.time() * 1000)
        if text == last_ai_entry and now - last_ai_entry_at < 1500:
            return
        last_ai_entry = text
        last_ai_entry_at = now
        print(f"[{get_ist_string()}] 🤖 AI: {text}")
        await add_entry({
            "type": "ai",
            "text": text,
            "timestamp": now,
            "istTime": get_ist_string()
        })

    async def add_function_entry(fn_name, args, result):
        await add_entry({
            "type": "function",
            "function": fn_name,
            "arguments": args,
            "result": result,
            "timestamp": int(time.time() * 1000),
            "istTime": get_ist_string()
        })

    async def play_hold_audio():
        if not _hold_audio_b64:
            return
        if plivo_closed:
            return
        try:
            await plivo_ws.send_text(json.dumps({
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": _hold_audio_b64
                }
            }))
            print(f"[HOLD] Hold audio played to caller")
        except Exception as e:
            print(f"[HOLD ERROR] type={type(e).__name__} msg={e}")

    async def play_transfer_audio():
        if not _transfer_audio_b64_global:
            return
        if plivo_closed:
            return
        try:
            await plivo_ws.send_text(json.dumps({
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": _transfer_audio_b64_global
                }
            }))
            print(f"[TRANSFER AUDIO] Transfer music played to caller")
        except Exception as e:
            print(f"[TRANSFER AUDIO ERROR] type={type(e).__name__} msg={e}")

    client = get_gemini_client(gemini_api_key)

    try:
        async for message in plivo_ws.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                start = data.get("start", {})
                stream_id = start.get("streamId")
                call_id = start.get("callId")
                print(f"📞 Call started: {call_id}")

                if call_id and call_id in active_calls:
                    call_info = active_calls[call_id]
                    request_uuid = call_id
                else:
                    now = time.time() * 1000
                    for uuid, info in active_calls.items():
                        if now - info.get("timestamp", 0) < 30000:
                            call_info = info
                            request_uuid = uuid
                            break

                if not request_uuid:
                    request_uuid = call_id or f"call_{int(time.time())}"
                    active_calls[request_uuid] = {
                        "phoneNumber": "unknown",
                        "isInbound": True,
                        "timestamp": time.time() * 1000
                    }
                    call_info = active_calls[request_uuid]

                caller_phone = resolve_caller_phone(data, "unknown")
                is_inbound = call_info.get("isInbound", True) if call_info else True
                campaign_data = call_info.get("campaignData") if call_info else None
                lead_data = call_info.get("leadData") if call_info else None
                resumption_handle = call_info.get("resumption_handle") if call_info else None
                if resumption_handle:
                    print(f"[RESUME] Restored resumption_handle for call_id={call_id} (returning from human transfer)")

                # ── Load caller memory ──
                caller_memory_doc = await load_caller_memory(caller_phone, db)
                memory_context = build_memory_context(caller_memory_doc)
                if caller_memory_doc:
                    print(f"[MEMORY] Returning caller: {caller_memory_doc.get('name')} "
                          f"(calls={caller_memory_doc.get('call_count', 1)})")

                print(f"[START] call_id={call_id} caller_phone={caller_phone} is_inbound={is_inbound}")
                break

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    audio_buffer.append(payload)

            elif event == "stop":
                print("📞 Call ended before start")
                await plivo_ws.close()
                return

    except Exception as e:
        print(f"[START ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        await plivo_ws.close()
        return

    async def plivo_receiver(session):
        nonlocal caller_phone, plivo_closed, speech_frame_count, last_interrupt_at, speech_active, speech_active_since, last_silence_at, speech_start_ms, ai_last_spoke_at

        try:
            await session.send_realtime_input(text="Hello")
        except Exception as e:
            print(f"[PLIVO GREETING ERROR] type={type(e).__name__} msg={e}")
            traceback.print_exc()
            raise

        for payload in audio_buffer:
            pcm16k = plivo_to_gemini(payload)
            try:
                await session.send_realtime_input(
                    audio=types.Blob(data=pcm16k, mime_type="audio/pcm;rate=16000")
                )
            except Exception as e:
                print(f"[PLIVO BUFFER-FLUSH ERROR] type={type(e).__name__} msg={e}")
                traceback.print_exc()
                raise
        audio_buffer.clear()

        try:
            async for message in plivo_ws.iter_text():
                data = json.loads(message)
                event = data.get("event")

                if event == "media":
                    payload = data.get("media", {}).get("payload")
                    if not payload:
                        continue
                    caller_phone = resolve_caller_phone(data, caller_phone)
                    pcm16k = plivo_to_gemini(payload)

                    speech_detected = await is_speech_silero(pcm16k)
                    now_ms = int(time.time() * 1000)

                    if speech_detected:
                        speech_frame_count += 1
                        last_silence_at = 0

                        if speech_start_ms == 0:
                            speech_start_ms = now_ms

                        speech_duration = now_ms - speech_start_ms

                        if not speech_active and speech_frame_count >= FRAMES_REQUIRED and speech_duration >= 100:
                            speech_active = True
                            speech_active_since = now_ms
                            if DEBUG:
                                print(f"[SILERO VAD] speech start detected (duration={speech_duration}ms)")
                            if now_ms - last_interrupt_at >= INTERRUPT_COOLDOWN:
                                if now_ms - ai_last_spoke_at < 1000:
                                    if DEBUG:
                                        print(f"[ECHO GATE] suppressed — within AI speaking window")
                                else:
                                    last_interrupt_at = now_ms
                                    if DEBUG:
                                        print(f"[SILERO VAD] interrupt signal (cooldown clear)")
                                    try:
                                        await plivo_ws.send_text(json.dumps({
                                            "event": "clearAudio",
                                            "stream_id": stream_id
                                        }))
                                        print(f"🛑 [{get_ist_string()}] [SILERO VAD] Barge-in → cleared Plivo audio")
                                    except Exception as clear_err:
                                        print(f"[SILERO VAD CLEAR ERROR] type={type(clear_err).__name__} msg={clear_err}")
                                    try:
                                        await session.send_realtime_input(
                                            activity_start=types.ActivityStart()
                                        )
                                        if DEBUG:
                                            print(f"[SILERO VAD] sent activity_start to Gemini")
                                    except Exception as activity_err:
                                        print(f"[SILERO VAD ACTIVITY_START ERROR] type={type(activity_err).__name__} msg={activity_err}")
                    else:
                        speech_frame_count = 0
                        speech_start_ms = 0
                        if speech_active:
                            if last_silence_at == 0:
                                last_silence_at = now_ms
                            elif now_ms - last_silence_at >= MIN_SPEECH_STOP_GAP_MS:
                                if now_ms - speech_active_since >= INTERRUPT_HOLD_MS:
                                    speech_active = False
                                    if DEBUG:
                                        print(f"[SILERO VAD] speech end detected (held {now_ms - speech_active_since}ms)")

                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm16k, mime_type="audio/pcm;rate=16000")
                        )
                    except Exception as send_err:
                        print(f"[PLIVO SEND ERROR] type={type(send_err).__name__} msg={send_err}")
                        traceback.print_exc()
                        raise

                elif event == "stop":
                    print("📞 Call ended (Plivo stop)")
                    plivo_closed = True
                    break

        except Exception as e:
            print(f"[PLIVO ERROR] type={type(e).__name__} msg={e}")
            traceback.print_exc()
            plivo_closed = True
            raise

    async def gemini_receiver(session):
        nonlocal resumption_handle, ai_last_spoke_at, plivo_closed, is_transferring, last_known_name, last_known_company, caller_info_saved
        event_counter = 0
        pending_calls = set()

        try:
            while True:
                got_event_this_pass = False

                async for response in session.receive():
                    got_event_this_pass = True
                    event_counter += 1

                    if os.getenv("DEBUG_FULL_DUMP") == "1":
                        print(f"[GEMINI RAW #{event_counter}] {dump_response_safe(response)}")

                    if response.go_away:
                        print("[GEMINI] ⚠️ GoAway received")
                        try:
                            await plivo_ws.send_text(json.dumps({
                                "event": "clearAudio",
                                "stream_id": stream_id
                            }))
                        except Exception as clear_err:
                            print(f"[GEMINI GOAWAY CLEAR ERROR] type={type(clear_err).__name__} msg={clear_err}")

                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            resumption_handle = update.new_handle

                    sc = response.server_content
                    if sc:
                        if sc.interrupted:
                            barge_time = get_ist_string()
                            try:
                                await plivo_ws.send_text(json.dumps({
                                    "event": "clearAudio",
                                    "stream_id": stream_id
                                }))
                                print(f"🛑 [{barge_time}] Barge-in → cleared Plivo audio")
                            except Exception as clear_err:
                                print(f"[BARGE-IN CLEAR ERROR] type={type(clear_err).__name__} msg={clear_err}")

                        if sc.input_transcription and sc.input_transcription.text:
                            await add_user_entry(sc.input_transcription.text)

                        if sc.output_transcription and sc.output_transcription.text:
                            await add_ai_entry(sc.output_transcription.text)

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    if plivo_closed:
                                        continue
                                    pcm24k = part.inline_data.data
                                    mulaw_b64 = gemini_to_plivo(pcm24k)
                                    try:
                                        ai_last_spoke_at = int(time.time() * 1000)
                                        await plivo_ws.send_text(json.dumps({
                                            "event": "playAudio",
                                            "media": {
                                                "contentType": "audio/x-mulaw",
                                                "sampleRate": 8000,
                                                "payload": mulaw_b64
                                            }
                                        }))
                                    except Exception as play_err:
                                        print(f"[PLAY AUDIO ERROR] type={type(play_err).__name__} msg={play_err}")
                                        plivo_closed = True
                                        break

                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            fn_name = fc.name
                            fn_args = dict(fc.args) if fc.args else {}
                            fn_id = fc.id

                            if DEBUG:
                                print(f"[TOOL CALL] {fn_name} args={fn_args}")

                            if fn_id in pending_calls:
                                print(f"[TOOL] Ignoring duplicate call id={fn_id} name={fn_name}")
                                continue
                            pending_calls.add(fn_id)

                            # ── STRICT: capture name/company from ANY tool call ──
                            incoming_name = fn_args.get("name")
                            incoming_company = fn_args.get("company")
                            if incoming_name and incoming_name != last_known_name:
                                last_known_name = incoming_name
                            if incoming_company and incoming_company != last_known_company:
                                last_known_company = incoming_company
                            if fn_name == "store_caller_info":
                                caller_info_saved = True
                            elif incoming_name and not caller_info_saved:
                                try:
                                    if DEBUG:
                                        print(f"[STRICT] Forcing store_caller_info — name seen via {fn_name}")
                                    await store_caller_info(
                                        name=last_known_name,
                                        company=last_known_company or "Not Provided",
                                        phone_number=caller_phone,
                                        request_uuid=request_uuid,
                                        inbound_lead_collection=inbound_lead_col
                                    )
                                    caller_info_saved = True
                                except Exception as force_err:
                                    print(f"[STRICT STORE ERROR] {force_err}")

                            # ── end_call ──
                            if fn_name == "end_call":
                                print(f"[END CALL] AI requested call termination (call_uuid={request_uuid})")
                                try:
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=fn_id,
                                                name=fn_name,
                                                response={"result": "Call ending."},
                                            )
                                        ]
                                    )
                                except Exception as e:
                                    print(f"[END CALL TOOL RESPONSE ERROR] {e}")

                                async def _delayed_hangup():
                                    nonlocal plivo_closed
                                    await asyncio.sleep(2.5)
                                    await hangup_call(request_uuid)
                                    plivo_closed = True
                                    try:
                                        await plivo_ws.close()
                                    except Exception:
                                        pass

                                asyncio.create_task(_delayed_hangup())
                                continue

                            # ── transfer_to_human ──
                            if fn_name == "transfer_to_human":
                                reason = fn_args.get("reason", "not specified")
                                print(f"[TRANSFER] AI requested human transfer (call_uuid={request_uuid}, reason={reason})")

                                active_calls.setdefault(request_uuid, {})
                                active_calls[request_uuid]["resumption_handle"] = resumption_handle
                                active_calls[request_uuid]["isInbound"] = is_inbound
                                active_calls[request_uuid]["campaignData"] = campaign_data
                                active_calls[request_uuid]["leadData"] = lead_data
                                active_calls[request_uuid]["phoneNumber"] = caller_phone
                                active_calls[request_uuid]["timestamp"] = time.time() * 1000

                                await add_function_entry(fn_name, fn_args, f"Escalating to human agent. Reason: {reason}")

                                try:
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=fn_id,
                                                name=fn_name,
                                                response={"result": "Transferring now."},
                                            )
                                        ]
                                    )
                                except Exception as e:
                                    print(f"[TRANSFER TOOL RESPONSE ERROR] {e}")

                                async def _delayed_transfer():
                                    nonlocal plivo_closed, is_transferring
                                    asyncio.create_task(play_transfer_audio())
                                    await asyncio.sleep(0.3)
                                    if PUBLIC_URL:
                                        await transfer_call(request_uuid, f"https://{PUBLIC_URL}/transfer-xml")
                                        is_transferring = True
                                    else:
                                        print("[TRANSFER ERROR] PUBLIC_URL not set in environment — cannot build transfer XML URL")
                                    plivo_closed = True

                                await _delayed_transfer()
                                return

                            asyncio.create_task(
                                background_tool(
                                    fn_id, fn_name, fn_args,
                                    caller_phone, request_uuid,
                                    appointments_col, inbound_lead_col,
                                    session, add_function_entry,
                                    pending_calls, db
                                )
                            )

                if not got_event_this_pass:
                    break

        except Exception as e:
            print(f"❌ [GEMINI RECEIVER CRASH] type={type(e).__name__} msg={e} after {event_counter} events")
            traceback.print_exc()
            raise

    for attempt in range(MAX_SESSION_RETRIES + 1):
        if plivo_closed:
            print("[RECONNECT] Plivo already closed, not reconnecting to Gemini")
            break

        config = build_gemini_config(is_inbound, campaign_data, lead_data, resumption_handle, memory_context)

        try:
            async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                print(f"✅ Gemini session connected (attempt {attempt})")
                await asyncio.gather(
                    plivo_receiver(session),
                    gemini_receiver(session)
                )
            break

        except Exception as e:
            print(f"[CONNECT ERROR] attempt={attempt} type={type(e).__name__} msg={e}")
            traceback.print_exc()

            if plivo_closed or is_transferring:
                print("[RECONNECT] Plivo connection is closed or call is transferring, stopping retries")
                break

            if attempt == MAX_SESSION_RETRIES:
                print("❌ [GIVING UP] Max Gemini reconnection attempts reached")
                break

            if not resumption_handle:
                print("⚠️ [RECONNECT] No resumption handle — reconnecting fresh")

            print(f"[RECONNECT] Retrying Gemini session in 1s (attempt {attempt + 1})...")
            await asyncio.sleep(1)
            continue

    print(f"[CLEANUP] request_uuid={request_uuid} is_transferring={is_transferring}")
    if request_uuid and not call_ended and not is_transferring:
        call_ended = True
        await update_call_end(
            request_uuid, is_inbound,
            db, inbound_col, outbound_col
        )

        # ── Save name/company immediately, independent of summary ──
        if caller_phone != "unknown":
            asyncio.create_task(save_caller_memory(
                phone_number=caller_phone,
                name=last_known_name,
                company=last_known_company,
                db=db
            ))

        async def save_summary_to_history():
            try:
                await generate_auto_summary(
                    request_uuid, is_inbound,
                    db, inbound_col, outbound_col,
                    broadcast_fn, gemini_api_key
                )
                col = inbound_col if is_inbound else outbound_col
                doc = await col.find_one({"requestUuid": request_uuid})
                summary_text = doc.get("summary", {}).get("text", "") if doc else ""
                if caller_phone != "unknown" and summary_text:
                    await save_call_summary(caller_phone, summary_text, db)
            except Exception as e:
                print(f"[SUMMARY ERROR] {e}")

        asyncio.create_task(save_summary_to_history())
        active_calls.pop(request_uuid, None)

    elif is_transferring:
        print(f"[CLEANUP SKIPPED] request_uuid={request_uuid} is being transferred to a human — state preserved in active_calls for resume")

    print("✅ Call cleanup done")
