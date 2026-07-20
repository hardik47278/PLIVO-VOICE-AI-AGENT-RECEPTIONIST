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

from fastapi import WebSocket
from google import genai
from google.genai import types

from iae_knowledge import IEI_KNOWLEDGE
from system_prompti import build_system_message, TOOL_DECLARATIONS
from transcript_service import (
    add_conversation_entry, update_call_end,
    generate_auto_summary, get_ist_string
)
from appointment_service import store_caller_info, book_appointment
from caller_memory import load_caller_memory, save_caller_memory, save_call_summary, build_memory_context

# ── RAG (vector search over ChromaDB via Gemini embeddings) ──
from rag_service import vector_search

DEBUG = os.getenv("DEBUG_TOOL_CALLS", "1") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
MAX_SESSION_RETRIES = 3

# ── Silero VAD constants ──
SILERO_THRESHOLD = 0.25
FRAMES_REQUIRED = 3
INTERRUPT_COOLDOWN = 1600
INTERRUPT_HOLD_MS = 400
MIN_SPEECH_STOP_GAP_MS = 350

# ── Filler audio (played while a tool call, e.g. RAG search, is running) ──
ENABLE_FILLER_AUDIO = os.getenv("ENABLE_FILLER_AUDIO", "0") == "1"
FILLER_AUDIO_PATH = os.getenv("FILLER_AUDIO_PATH", "")  # path to a raw 8kHz mulaw file, pre-encoded
FILLER_CHUNK_MS = 20  # Plivo expects small frames; 20ms @ 8kHz mulaw = 160 bytes
_filler_chunks_cache = None

active_calls = {}
stream_call_map = {}

_gemini_client = None

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
    #pcm8k = audioop.mul(pcm8k, 2, 3.0)
    mulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(mulaw).decode()


def _load_filler_chunks():
    """
    Loads a pre-recorded, pre-encoded 8kHz mulaw filler clip from disk and
    splits it into small base64 chunks Plivo can consume as playAudio frames.
    Cached after first load. Returns [] if no filler file configured/found.
    """
    global _filler_chunks_cache
    if _filler_chunks_cache is not None:
        return _filler_chunks_cache

    if not FILLER_AUDIO_PATH or not os.path.exists(FILLER_AUDIO_PATH):
        _filler_chunks_cache = []
        return _filler_chunks_cache

    try:
        with open(FILLER_AUDIO_PATH, "rb") as f:
            raw = f.read()

        bytes_per_chunk = int(8000 * (FILLER_CHUNK_MS / 1000))
        chunks = [
            base64.b64encode(raw[i:i + bytes_per_chunk]).decode()
            for i in range(0, len(raw), bytes_per_chunk)
            if raw[i:i + bytes_per_chunk]
        ]
        _filler_chunks_cache = chunks
        print(f"[FILLER] Loaded {len(chunks)} chunks from {FILLER_AUDIO_PATH}")
    except Exception as e:
        print(f"[FILLER] Failed to load filler audio: {type(e).__name__}: {e}")
        _filler_chunks_cache = []

    return _filler_chunks_cache


async def play_filler_audio(plivo_ws: WebSocket, stream_id: str):
    """
    Loops a short pre-recorded clip into the Plivo stream while a slow tool call
    (e.g. vector search) is running. Cancelled as soon as the real result is ready.
    """
    chunks = _load_filler_chunks()
    if not chunks:
        return
    try:
        while True:
            for chunk in chunks:
                await plivo_ws.send_text(json.dumps({
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": chunk
                    }
                }))
                await asyncio.sleep(FILLER_CHUNK_MS / 1000)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[FILLER] error during playback: {type(e).__name__}: {e}")


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
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
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
            print(f"[TOOL] query_company_data (RAG) topic={topic}")
        return await vector_search(topic)

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

    return "Tool not found."


async def handle_media_stream(
    plivo_ws: WebSocket,
    db,
    inbound_col,
    outbound_col,
    appointments_col,
    inbound_lead_col,
    broadcast_fn=None,
    gemini_api_key=None
):
    await plivo_ws.accept()
    print("✅ Plivo connected")

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

    memory_context = ""
    last_known_name = ""
    last_known_company = ""

    speech_frame_count = 0
    last_interrupt_at = 0
    speech_active = False
    speech_active_since = 0
    last_silence_at = 0
    speech_start_ms = 0

    ai_last_spoke_at = 0

    if _silero_model is not None:
        print(f"✅ Silero VAD ready (threshold={SILERO_THRESHOLD})")
    else:
        print("⚠️ Silero VAD not available — barge-in VAD disabled. Run: pip install torch && torch.hub.load snakers4/silero-vad")

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
                if caller_phone == "unknown" and call_info:
                    stored_phone = call_info.get("phoneNumber","unknown")
                    if stored_phone != "unknown":
                        caller_phone = stored_phone
                is_inbound = call_info.get("isInbound", True) if call_info else True
                campaign_data = call_info.get("campaignData") if call_info else None
                lead_data = call_info.get("leadData") if call_info else None

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
        nonlocal resumption_handle, ai_last_spoke_at, last_known_name, last_known_company
        event_counter = 0

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
                                        traceback.print_exc()
                                        break

                    if response.tool_call:
                        function_responses = []

                        # ── Optional: play filler/hold audio while tool calls
                        # (e.g. RAG vector search) are in progress. Only starts
                        # if ENABLE_FILLER_AUDIO=1 and a filler clip is configured.
                        filler_task = None
                        if ENABLE_FILLER_AUDIO:
                            filler_task = asyncio.create_task(
                                play_filler_audio(plivo_ws, stream_id)
                            )

                        try:
                            for fc in response.tool_call.function_calls:
                                fn_name = fc.name
                                fn_args = dict(fc.args) if fc.args else {}
                                fn_id = fc.id

                                if DEBUG:
                                    print(f"[TOOL CALL] {fn_name} args={fn_args}")

                                if fn_name == "store_caller_info":
                                    last_known_name = fn_args.get("name", last_known_name)
                                    last_known_company = fn_args.get("company", last_known_company)

                                try:
                                    result = await execute_tool(
                                        fn_name, fn_args,
                                        caller_phone, request_uuid,
                                        appointments_col, inbound_lead_col,
                                        db
                                    )
                                except Exception as tool_err:
                                    print(f"[TOOL EXEC ERROR] name={fn_name} type={type(tool_err).__name__} msg={tool_err}")
                                    traceback.print_exc()
                                    result = f"Error: {tool_err}"

                                await add_function_entry(fn_name, fn_args, result)

                                function_responses.append(
                                    types.FunctionResponse(
                                        id=fn_id,
                                        name=fn_name,
                                        response={"result": result}
                                    )
                                )
                        finally:
                            if filler_task is not None:
                                filler_task.cancel()
                                try:
                                    await filler_task
                                except asyncio.CancelledError:
                                    pass

                        try:
                            await session.send_tool_response(
                                function_responses=function_responses
                            )
                        except Exception as tr_err:
                            print(f"[TOOL RESPONSE SEND ERROR] type={type(tr_err).__name__} msg={tr_err}")
                            traceback.print_exc()
                            raise

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

            if plivo_closed:
                print("[RECONNECT] Plivo connection is closed, stopping retries")
                break

            if attempt == MAX_SESSION_RETRIES:
                print("❌ [GIVING UP] Max Gemini reconnection attempts reached")
                break

            if not resumption_handle:
                print("⚠️ [RECONNECT] No resumption handle available — reconnecting fresh, conversation context will be lost for this segment")

            print(f"[RECONNECT] Retrying Gemini session in 1s (attempt {attempt + 1})...")
            await asyncio.sleep(1)
            continue

    print(f"[CLEANUP] request_uuid={request_uuid}")
    if request_uuid and not call_ended:
        call_ended = True
        await update_call_end(
            request_uuid, is_inbound,
            db, inbound_col, outbound_col
        )

        if caller_phone != "unknown":
            asyncio.create_task(save_caller_memory(
                phone_number=caller_phone,
                name=last_known_name,
                company=last_known_company,
                db=db
            ))

        async def save_memory_with_summary():
            summary_text = ""
            try:
                await generate_auto_summary(
                    request_uuid, is_inbound,
                    db, inbound_col, outbound_col,
                    broadcast_fn, gemini_api_key
                )
                col = inbound_col if is_inbound else outbound_col
                doc = await col.find_one({"requestUuid": request_uuid})
                summary_text = doc.get("summary", "") if doc else ""
            except Exception as e:
                print(f"[SUMMARY ERROR] {e} — saving memory without summary")

            try:
                if caller_phone != "unknown" and summary_text:
                    await save_call_summary(caller_phone, summary_text, db)
            except Exception as e:
                print(f"[MEMORY SAVE ERROR] {e}")

        asyncio.create_task(save_memory_with_summary())
        active_calls.pop(request_uuid, None)

    print("✅ Call cleanup done")