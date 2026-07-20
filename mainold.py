import asyncio
import os
import time

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from qq_new_humna import handle_media_stream, active_calls
from caller_memory import ensure_caller_memory_index

import plivo


# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------
load_dotenv()

app = FastAPI()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL1 = os.getenv("GEMINI_MODEL")

MONGO_URI = (
    "mongodb+srv://voiceagent:Test123456@cluster0.1ijo6.mongodb.net/"
    "workmates_voice?retryWrites=true&w=majority"
)

PUBLIC_URL = "ff56-59-144-30-58.ngrok-free.app"

HUMAN_AGENT_NUMBER = os.getenv("HUMAN_AGENT_NUMBER")


# ---------------------------------------------------------
# Static Files
# ---------------------------------------------------------
os.makedirs("static", exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ---------------------------------------------------------
# MongoDB
# ---------------------------------------------------------
mongo_client = AsyncIOMotorClient(MONGO_URI)

db = mongo_client["workmates_voice"]

inbound_col = db["inbound_transcripts"]
outbound_col = db["call_transcripts"]

appointments_col = db["appointments"]

inbound_lead_col = db["inbound_leads"]


# ---------------------------------------------------------
# Broadcast Function
# ---------------------------------------------------------
def broadcast_fn(request_uuid, entry, call_type):
    pass


# ---------------------------------------------------------
# Incoming Call Webhook
# ---------------------------------------------------------
@app.post("/incoming-call")
async def incoming_call(request: Request):

    body = await request.form()

    call_id = body.get("CallUUID") or body.get("callId") or ""

    from_number = (
        body.get("From")
        or body.get("from")
        or "unknown"
    )

    print(
        f"📞 Incoming call: {call_id} "
        f"from {from_number}"
    )

    active_calls[call_id] = {
        "phoneNumber": from_number,
        "isInbound": True,
        "timestamp": time.time() * 1000,
        "campaignData": None,
        "leadData": None
    }

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream
        keepCallAlive="true"
        bidirectional="true"
        contentType="audio/x-mulaw;rate=8000">

        wss://{PUBLIC_URL}/media-stream

    </Stream>
</Response>"""

    print(
        f"📤 Stream URL: "
        f"wss://{PUBLIC_URL}/media-stream"
    )

    return PlainTextResponse(
        content=xml,
        media_type="text/xml"
    )


# ---------------------------------------------------------
# Media WebSocket
# ---------------------------------------------------------
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):

    await handle_media_stream(
        plivo_ws=websocket,
        db=db,
        inbound_col=inbound_col,
        outbound_col=outbound_col,
        appointments_col=appointments_col,
        inbound_lead_col=inbound_lead_col,
        broadcast_fn=broadcast_fn,
        gemini_api_key=GEMINI_API_KEY
    )


# ---------------------------------------------------------
# Transfer XML
# ---------------------------------------------------------
@app.api_route(
    "/transfer-xml",
    methods=["GET", "POST"]
)
async def transfer_xml():

    if not HUMAN_AGENT_NUMBER:
        print(
            "⚠️ [TRANSFER-XML] "
            "HUMAN_AGENT_NUMBER not set"
        )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>

    <Dial
        action="https://{PUBLIC_URL}/after-human-call"
        method="POST">

        <Number>
            {HUMAN_AGENT_NUMBER}
        </Number>

    </Dial>

</Response>"""

    print(
        "[TRANSFER-XML] "
        f"agent={HUMAN_AGENT_NUMBER}"
    )

    return PlainTextResponse(
        content=xml,
        media_type="text/xml"
    )


# ---------------------------------------------------------
# After Human Call
# ---------------------------------------------------------
@app.post("/after-human-call")
async def after_human_call(request: Request):

    body = await request.form()

    call_uuid = body.get("CallUUID")

    dial_status = body.get("DialStatus")

    saved = active_calls.get(call_uuid)

    print(
        f"[AFTER-HUMAN-CALL] "
        f"call_uuid={call_uuid} "
        f"dial_status={dial_status} "
        f"has_saved_state={bool(saved)} "
        f"has_resumption_handle="
        f"{bool(saved and saved.get('resumption_handle'))}"
    )

    if (
        dial_status == "completed"
        and saved
        and saved.get("resumption_handle")
    ):

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>

    <Stream
        keepCallAlive="true"
        bidirectional="true"
        contentType="audio/x-mulaw;rate=8000">

        wss://{PUBLIC_URL}/media-stream

    </Stream>

</Response>"""

        print(
            "[AFTER-HUMAN-CALL] "
            f"Reconnecting AI "
            f"for {call_uuid}"
        )

    else:

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>

    <Speak>
        Thank you for calling
        The Institution of Engineers India.
        Have a great day!
    </Speak>

    <Hangup/>

</Response>"""

        print(
            "[AFTER-HUMAN-CALL] "
            f"Ending call "
            f"dial_status={dial_status}"
        )

    return PlainTextResponse(
        content=xml,
        media_type="text/xml"
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
async def health():

    return {
        "status": "ok",
        "active_calls": len(active_calls)
    }


# ---------------------------------------------------------
# Startup Tasks
# ---------------------------------------------------------
@app.on_event("startup")
async def startup():

    try:
        await ensure_caller_memory_index()

    except Exception as e:
        print("Caller memory index error:", e)


# ---------------------------------------------------------
# Local Run
# ---------------------------------------------------------
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )