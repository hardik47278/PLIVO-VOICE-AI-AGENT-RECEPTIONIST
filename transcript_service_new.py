# transcript_service.py
import asyncio
from datetime import datetime, timezone
import pytz
from bson import ObjectId

IST = pytz.timezone("Asia/Kolkata")

def get_ist_now():
    return datetime.now(IST)

def get_ist_string():
    return get_ist_now().strftime("%d/%m/%Y, %I:%M:%S %p")

def get_collection(is_inbound, inbound_col, outbound_col):
    return inbound_col if is_inbound else outbound_col

async def create_transcript(request_uuid, phone_number, is_inbound, db, inbound_col, outbound_col, lead_id=None, campaign_id=None):
    col = get_collection(is_inbound, inbound_col, outbound_col)
    if col is None:
        return None
    try:
        existing = await col.find_one({"requestUuid": request_uuid})
        if existing:
            return existing

        transcript = {
            "requestUuid": request_uuid,
            "phoneNumber": phone_number or "Unknown",
            "leadId": ObjectId(lead_id) if lead_id else None,
            "campaignId": ObjectId(campaign_id) if campaign_id else None,
            "startTime": datetime.utcnow().timestamp() * 1000,
            "endTime": None,
            "duration": None,
            "conversation": [],
            "summary": None,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "istStartTime": get_ist_string(),
            "status": "active",
            "callType": "inbound" if is_inbound else "outbound"
        }
        result = await col.insert_one(transcript)
        print(f"✅ Transcript created for {request_uuid}")
        return {**transcript, "_id": result.inserted_id}
    except Exception as e:
        print(f"❌ Error creating transcript: {e}")
        return None

async def _write_conversation_entry(request_uuid, entry, is_inbound, db, inbound_col, outbound_col, broadcast_fn=None):
    """Actual DB write — runs as a background task so it never blocks the WebSocket receive loop."""
    col = get_collection(is_inbound, inbound_col, outbound_col)
    if col is None:
        if broadcast_fn:
            broadcast_fn(request_uuid, entry, "inbound" if is_inbound else "outbound")
        return
    try:
        result = await col.update_one(
            {"requestUuid": request_uuid},
            {"$push": {"conversation": entry}, "$set": {"updatedAt": datetime.utcnow()}}
        )
        if result.matched_count == 0:
            # Transcript doesn't exist yet — create it, then push this entry
            created = await create_transcript(request_uuid, "Unknown", is_inbound, db, inbound_col, outbound_col)
            if created:
                await col.update_one(
                    {"requestUuid": request_uuid},
                    {"$push": {"conversation": entry}, "$set": {"updatedAt": datetime.utcnow()}}
                )

        if broadcast_fn:
            broadcast_fn(request_uuid, entry, "inbound" if is_inbound else "outbound")
    except Exception as e:
        print(f"❌ Error adding entry: {e}")

async def add_conversation_entry(request_uuid, entry, is_inbound, db, inbound_col, outbound_col, broadcast_fn=None):
    """
    Fire-and-forget the DB write so it never blocks the caller (the WebSocket
    receive loop in websocket_handler.py). This is critical during rapid-fire
    transcript streaming (Gemini sends many small text chunks per second) —
    awaiting a Mongo round-trip per chunk can stall the event loop long enough
    to miss WebSocket keepalive pings and trigger a 1011 timeout disconnect.
    """
    asyncio.create_task(
        _write_conversation_entry(request_uuid, entry, is_inbound, db, inbound_col, outbound_col, broadcast_fn)
    )

async def update_call_end(request_uuid, is_inbound, db, inbound_col, outbound_col):
    col = get_collection(is_inbound, inbound_col, outbound_col)
    if col is None:
        return None
    try:
        transcript = await col.find_one({"requestUuid": request_uuid})
        if transcript:
            end_time = datetime.utcnow().timestamp() * 1000
            duration = end_time - transcript.get("startTime", end_time)
            await col.update_one(
                {"requestUuid": request_uuid},
                {"$set": {
                    "endTime": end_time,
                    "duration": duration,
                    "updatedAt": datetime.utcnow(),
                    "istEndTime": get_ist_string(),
                    "status": "completed"
                }}
            )
            print(f"✅ Call ended: {request_uuid}, duration: {duration:.0f}ms")
            return await col.find_one({"requestUuid": request_uuid})
        return None
    except Exception as e:
        print(f"❌ Error updating call end: {e}")
        return None

async def generate_auto_summary(request_uuid, is_inbound, db, inbound_col, outbound_col, broadcast_fn=None, gemini_api_key=None):
    print(f"🤖 Generating summary for {request_uuid}")
    col = get_collection(is_inbound, inbound_col, outbound_col)
    if col is None:
        return

    await asyncio.sleep(3)

    try:
        transcript = await col.find_one({"requestUuid": request_uuid})
        if not transcript or not transcript.get("conversation"):
            print(f"❌ No conversation found for {request_uuid}")
            return

        conversation_text = "\n".join([
            f"{'Caller' if e['type'] == 'user' else 'Assistant'}: {e.get('text', '')}"
            for e in transcript["conversation"]
            if e.get("text")
        ])

        summary_data = {
            "text": "Call completed. See transcript for details.",
            "actionItems": [],
            "insights": "Review transcript for insights.",
            "mood": "Neutral",
            "style": "auto-generated",
            "generatedAt": datetime.utcnow().timestamp() * 1000,
            "generatedAtIST": get_ist_string(),
            "callType": "inbound" if is_inbound else "outbound"
        }

        # Use Gemini for summary if API key provided
        if gemini_api_key:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}",
                        json={
                            "contents": [{
                                "parts": [{
                                    "text": f"""Analyze this call and return JSON only:
{{
  "summary": "2-3 sentence overview",
  "actionItems": ["action1", "action2"],
  "insights": "key insights",
  "mood": "Positive or Neutral or Negative"
}}

Call transcript:
{conversation_text}"""
                                }]
                            }],
                            "generationConfig": {"temperature": 0.3}
                        },
                        timeout=30
                    )
                    result = response.json()
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    import json, re
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                        summary_data.update({
                            "text": analysis.get("summary", summary_data["text"]),
                            "actionItems": analysis.get("actionItems", []),
                            "insights": analysis.get("insights", ""),
                            "mood": analysis.get("mood", "Neutral"),
                            "model": "gemini-2.0-flash"
                        })
                        print(f"✅ Summary generated via Gemini")
            except Exception as e:
                print(f"⚠️ Gemini summary error: {e}")

        await col.update_one(
            {"requestUuid": request_uuid},
            {"$set": {"summary": summary_data, "updatedAt": datetime.utcnow()}}
        )

        if broadcast_fn:
            broadcast_fn(request_uuid, {
                "type": "summary",
                "text": summary_data["text"],
                "callType": "inbound" if is_inbound else "outbound"
            }, "inbound" if is_inbound else "outbound")

        print(f"🎉 Summary stored for {request_uuid}")
    except Exception as e:
        print(f"❌ Summary error: {e}")