# websocket_handler.py
import asyncio
import base64
import json
import time
import os
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from fastapi import WebSocket
import websockets

from realtime_config import GEMINI_WS_URL
from workmates_knowledge import WORKMATES_KNOWLEDGE
from system_prompt import build_system_message, TOOL_DECLARATIONS
from transcript_service import (
    add_conversation_entry, update_call_end,
    generate_auto_summary, get_ist_string
)
from appointment_service import store_caller_info, book_appointment

DEBUG = os.getenv("DEBUG_TOOL_CALLS", "1") == "1"

active_calls = {}
stream_call_map = {}


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



def plivo_to_gemini(payload: str) -> bytes:
    mulaw = base64.b64decode(payload)
    pcm8k = audioop.ulaw2lin(mulaw, 2)
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return pcm16k


def gemini_to_plivo(pcm24k: bytes) -> str:
    pcm8k, _ = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, None)
    pcm8k = audioop.mul(pcm8k,2,3.0)
    mulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(mulaw).decode()


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

    # ── Connect to Gemini ──────────────────────────────────
    try:
        gemini_ws = await websockets.connect(GEMINI_WS_URL)
        print("✅ Gemini connected")
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        await plivo_ws.close()
        return

    call_info = None
    campaign_data = None
    lead_data = None

    # ── Send Gemini setup ──────────────────────────────────
    try:
        await gemini_ws.send(json.dumps({
            "setup": {
                "model": f"models/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-native-audio-preview-12-2025')}",

                        "systemInstruction": {
            "parts": [{
                "text": """
                You are a professional appointment booking assistant.

                Speak in Indian English with a natural Indian accent.
                Use Indian pronunciation for names, dates, and numbers.
                Avoid American slang and expressions.
                If the user speaks Hindi, respond in natural Hindi.
                If the user speaks Bengali, respond in natural Bengali.
                """
            }]
        },




                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "temperature": 0.4,
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede"
                            }
                        }
                    }
                },
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                        "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                        "silenceDurationMs": 500
                    }
                },
                "systemInstruction": {
                    "parts": [{"text": build_system_message(is_inbound, campaign_data, lead_data)}]
                },
                "tools": [
    {
        "functionDeclarations": TOOL_DECLARATIONS
    },
    
]
                
            }
        }))
        print("✅ Gemini setup sent")
    except Exception as e:
        print(f"❌ Gemini setup failed: {e}")
        await plivo_ws.close()
        return

    # ── Plivo → Gemini ─────────────────────────────────────
    async def plivo_receiver():
        nonlocal request_uuid, caller_phone, is_inbound, stream_id, call_ended, call_info, campaign_data, lead_data
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

                    await gemini_ws.send(json.dumps({
                        "realtimeInput": {
                            "text": "Hello"
                        }
                    }))

                elif event == "media":
                    payload = data.get("media", {}).get("payload")
                    if not payload:
                        continue
                    caller_phone = resolve_caller_phone(data, caller_phone)
                    pcm16k = plivo_to_gemini(payload)
                    await gemini_ws.send(json.dumps({
                        "realtimeInput": {
                            "audio": {
                                "data": base64.b64encode(pcm16k).decode(),
                                "mimeType": "audio/pcm;rate=16000"
                            }
                        }
                    }))

                elif event == "stop":
                    print("📞 Call ended (Plivo stop)")
                    break

        except Exception as e:
            print(f"plivo_receiver error: {e}")

    # ── Gemini → Plivo ─────────────────────────────────────
    async def gemini_receiver():
        nonlocal call_ended
        try:
            async for message in gemini_ws:
                response = json.loads(message)

                if "serverContent" in response:
                    sc = response["serverContent"]

                    # barge-in
                    if sc.get("interrupted"):
                        try:
                            await plivo_ws.send_text(json.dumps({
                                "event": "clearAudio",
                                "stream_id": stream_id
                            }))
                            print("🛑 Barge-in → cleared Plivo audio")
                        except Exception:
                            pass

                    # user transcript
                    if "inputTranscription" in sc:
                        text = sc["inputTranscription"].get("text", "").strip()
                        if text:
                            await add_user_entry(text)

                    # AI transcript
                    if "outputTranscription" in sc:
                        text = sc["outputTranscription"].get("text", "").strip()
                        if text:
                            await add_ai_entry(text)

                    # audio chunks
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for part in parts:
                        if "inlineData" in part:
                            pcm24k = base64.b64decode(part["inlineData"]["data"])
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
                            except Exception:
                                break

                # tool calls
                if "toolCall" in response:
                    function_responses = []
                    for fc in response["toolCall"].get("functionCalls", []):
                        fn_name = fc.get("name")
                        fn_args = fc.get("args", {})
                        fn_id = fc.get("id")

                        if DEBUG:
                            print(f"[TOOL CALL] {fn_name} args={fn_args}")

                        result = await execute_tool(
                            fn_name, fn_args,
                            caller_phone, request_uuid,
                            appointments_col, inbound_lead_col
                        )

                        await add_function_entry(fn_name, fn_args, result)

                        function_responses.append({
                            "name": fn_name,
                            "id": fn_id,
                            "response": {"result": result}
                        })

                    await gemini_ws.send(json.dumps({
                        "toolResponse": {
                            "functionResponses": function_responses
                        }
                    }))

        except Exception as e:
            print(f"gemini_receiver error: {e}")

    # ── Run both directions ─────────────────────────────────
    try:
        await asyncio.gather(plivo_receiver(), gemini_receiver())
    finally:
        try:
            await gemini_ws.close()
        except Exception:
            pass

        if request_uuid and not call_ended:
            call_ended = True
            await update_call_end(request_uuid, is_inbound, db, inbound_col, outbound_col)
            asyncio.create_task(
                generate_auto_summary(
                    request_uuid, is_inbound,
                    db, inbound_col, outbound_col,
                    broadcast_fn, gemini_api_key
                )
            )
            active_calls.pop(request_uuid, None)

        print("✅ Call cleanup done")