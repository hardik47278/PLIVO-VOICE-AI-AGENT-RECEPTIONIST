# websocket_handler.py
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

from fastapi import WebSocket
from google import genai
from google.genai import types

from workmates_knowledge import WORKMATES_KNOWLEDGE
from system_prompt import build_system_message, TOOL_DECLARATIONS
from transcript_service import (
    add_conversation_entry, update_call_end,
    generate_auto_summary, get_ist_string
)
from appointment_service import store_caller_info, book_appointment

DEBUG = os.getenv("DEBUG_TOOL_CALLS", "1") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
MAX_SESSION_RETRIES = 3

active_calls = {}
stream_call_map = {}

# ── Pre-build client at module level ──────────────────────
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
    """Best-effort full dump of a Gemini Live response object for debugging."""
    try:
        if hasattr(response, "model_dump"):
            d = response.model_dump(exclude_none=True)
            return json.dumps(d, default=str)[:2000]  # cap length, avoid flooding terminal with huge audio b64
    except Exception as dump_err:
        return f"<dump failed: {type(dump_err).__name__}: {dump_err}> repr={repr(response)[:500]}"
    return repr(response)[:500]


# ── Audio conversion ───────────────────────────────────────
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


# ── Pre-build function declarations once ──────────────────
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


def build_gemini_config(is_inbound=True, campaign_data=None, lead_data=None, resumption_handle=None):
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        temperature=0.4,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"
                )
            )
        ),
      
      
        system_instruction=types.Content(
            parts=[types.Part(
                text=build_system_message(is_inbound, campaign_data, lead_data)
            )]
        ),
        tools=[types.Tool(function_declarations=FUNCTION_DECLARATIONS)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # ── Session resumption: lets us reconnect mid-call if Google
        # resets the underlying WebSocket, instead of the call dying ──
        session_resumption=types.SessionResumptionConfig(
            handle=resumption_handle
        ),
        # ── Context window compression: extends the 15-min audio-only
        # session cap to effectively unlimited duration ──────────────
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    )


# ── Tool executor ──────────────────────────────────────────
async def execute_tool(name, args, caller_phone, request_uuid, appointments_col, inbound_lead_col):
    if name == "query_company_data":
        topic = args.get("topic", "")
        if DEBUG:
            print(f"[TOOL] query_company_data topic={topic}")
        return WORKMATES_KNOWLEDGE if topic == "workmates" else "No data available."

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
        return result

    return "Tool not found."


# ── Main WebSocket handler ─────────────────────────────────
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
    plivo_closed = False  # set True once Plivo sends "stop" or socket dies

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

    # ── Step 1: Wait for Plivo start event, buffer any early audio ────
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

    # ── Plivo → Gemini ─────────────────────────────────────
    async def plivo_receiver(session):
        nonlocal caller_phone, plivo_closed

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
            raise  # propagate so the outer connect loop knows this session is done

    # ── Gemini → Plivo ─────────────────────────────────────
    async def gemini_receiver(session):
        nonlocal resumption_handle
        event_counter = 0

        try:
            # ── FIX (Issue #1224 pattern) ───────────────────────────
            # session.receive() returns an async generator that the SDK
            # silently exhausts once per "turn" (e.g. right after
            # turn_complete), even though the underlying connection is
            # still alive. If we only consume it once, this coroutine
            # returns and nobody reads Gemini's output ever again for
            # the rest of the call — audio keeps flowing from Plivo
            # into the void until the connection eventually times out.
            #
            # Fix: keep re-entering session.receive() in an outer loop
            # instead of treating generator-exhaustion as session-end.
            # Only actually stop if a pass yields zero events (real
            # sign the session is dead) or an exception is raised.
            while True:
                got_event_this_pass = False

                async for response in session.receive():
                    got_event_this_pass = True
                    event_counter += 1

                    if os.getenv("DEBUG_FULL_DUMP") == "1":
                        print(f"[GEMINI RAW #{event_counter}] {dump_response_safe(response)}")

                    # GoAway: server is about to terminate the connection.
                    if response.go_away:
                        print("[GEMINI] ⚠️ GoAway received")
                        try:
                            await plivo_ws.send_text(json.dumps({
                                "event": "clearAudio",
                                "stream_id": stream_id
                            }))
                        except Exception as clear_err:
                            print(f"[GEMINI GOAWAY CLEAR ERROR] type={type(clear_err).__name__} msg={clear_err}")

                    # Capture the latest resumption handle so we can reconnect
                    # seamlessly if this session gets reset.
                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            resumption_handle = update.new_handle

                    sc = response.server_content
                    if sc:
                        if sc.interrupted:
                            try:
                                await plivo_ws.send_text(json.dumps({
                                    "event": "clearAudio",
                                    "stream_id": stream_id
                                }))
                                print("🛑 Barge-in → cleared Plivo audio")
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
                        for fc in response.tool_call.function_calls:
                            fn_name = fc.name
                            fn_args = dict(fc.args) if fc.args else {}
                            fn_id = fc.id

                            if DEBUG:
                                print(f"[TOOL CALL] {fn_name} args={fn_args}")

                            try:
                                result = await execute_tool(
                                    fn_name, fn_args,
                                    caller_phone, request_uuid,
                                    appointments_col, inbound_lead_col
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

                        try:
                            await session.send_tool_response(
                                function_responses=function_responses
                            )
                        except Exception as tr_err:
                            print(f"[TOOL RESPONSE SEND ERROR] type={type(tr_err).__name__} msg={tr_err}")
                            traceback.print_exc()
                            raise

                # The inner `async for` generator exhausted. This is the
                # known SDK quirk — it does NOT necessarily mean the
                # connection is dead. Loop back and call receive() again.
                if not got_event_this_pass:
                    # We got literally nothing this pass — the session is
                    # genuinely gone (or never produced anything). Stop
                    # spinning in a tight empty loop.
                    break

        except Exception as e:
            print(f"❌ [GEMINI RECEIVER CRASH] type={type(e).__name__} msg={e} after {event_counter} events")
            traceback.print_exc()
            raise  # propagate so the outer connect loop knows this session is done

    # ── Connect to Gemini with retry + session resumption ─────────────
    for attempt in range(MAX_SESSION_RETRIES + 1):
        if plivo_closed:
            print("[RECONNECT] Plivo already closed, not reconnecting to Gemini")
            break

        config = build_gemini_config(is_inbound, campaign_data, lead_data, resumption_handle)

        try:
            async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                print(f"✅ Gemini session connected (attempt {attempt})")
                await asyncio.gather(
                    plivo_receiver(session),
                    gemini_receiver(session)
                )
            # Clean exit (e.g. Plivo sent "stop") — don't retry
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

    # ── Cleanup ─────────────────────────────────────────────
    print(f"[CLEANUP] request_uuid={request_uuid}")
    if request_uuid and not call_ended:
        call_ended = True
        await update_call_end(
            request_uuid, is_inbound,
            db, inbound_col, outbound_col
        )
        asyncio.create_task(
            generate_auto_summary(
                request_uuid, is_inbound,
                db, inbound_col, outbound_col,
                broadcast_fn, gemini_api_key
            )
        )
        active_calls.pop(request_uuid, None)

    print("✅ Call cleanup done")