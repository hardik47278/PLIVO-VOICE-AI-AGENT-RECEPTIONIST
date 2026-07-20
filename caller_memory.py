# caller_memory.py
from datetime import datetime


async def load_caller_memory(phone_number: str, db):
    """
    Load caller profile by phone number.
    Returns None if first-time caller.
    """
    if not phone_number or phone_number == "unknown":
        return None

    collection = db["caller_memory"]
    try:
        doc = await collection.find_one({"phone_number": phone_number})
        return doc
    except Exception as e:
        print(f"[MEMORY LOAD ERROR] {e}")
        return None


async def save_caller_memory(phone_number: str, name: str, company: str, db):
    """
    Save/update caller profile after a call.
    Increments call_count, updates last_call timestamp.
    """
    if not phone_number or phone_number == "unknown":
        return

    collection = db["caller_memory"]
    try:
        await collection.update_one(
            {"phone_number": phone_number},
            {
                "$set": {
                    "phone_number": phone_number,
                    "name": name or "Unknown",
                    "company": company or "Not Provided",
                    "last_call": datetime.utcnow().isoformat()
                },
                "$inc": {"call_count": 1}
            },
            upsert=True
        )
        print(f"[MEMORY] Saved profile for {phone_number}")
    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")


async def save_call_summary(phone_number: str, summary, db):
    """
    Append a short call summary to caller history.
    Keeps only last 5 summaries to control growth.
    """
    if not phone_number or phone_number == "unknown" or not summary:
        return

    # ── FIX: coerce to string before slicing ────────────────
    # summary can arrive as None, a dict, or other non-string
    # type depending on what generate_auto_summary returned
    # upstream. summary[:200] on None throws
    # "slice(None, 200, None)" — this guard prevents that and
    # also drops empty/whitespace-only summaries.
    summary = str(summary).strip()
    if not summary or summary.lower() in ("none", "null"):
        return

    collection = db["caller_memory"]
    try:
        await collection.update_one(
            {"phone_number": phone_number},
            {
                "$push": {
                    "call_history": {
                        "$each": [{
                            "timestamp": datetime.utcnow().isoformat(),
                            "summary": summary[:200]  # cap length
                        }],
                        "$slice": -5  # keep only last 5 entries
                    }
                }
            },
            upsert=True
        )
    except Exception as e:
        print(f"[SUMMARY SAVE ERROR] {e}")


async def ensure_caller_memory_index(db):
    """
    Call once at app startup to create index for fast lookups.
    """
    try:
        await db["caller_memory"].create_index("phone_number", unique=True)
        print("✅ caller_memory index created")
    except Exception as e:
        print(f"[INDEX ERROR] {e}")


def build_memory_context(caller_memory: dict) -> str:
    """
    Build a terse memory context string for system prompt injection.
    Returns empty string if no memory exists.
    """
    if not caller_memory:
        return ""

    name = caller_memory.get("name", "")
    company = caller_memory.get("company", "")
    call_count = caller_memory.get("call_count", 1)
    history = caller_memory.get("call_history", [])

    recent = history[-2:] if history else []
    summaries = " | ".join([h.get("summary", "")[:80] for h in recent])

    context = f"""
## RETURNING CALLER CONTEXT
Name: {name} | Status: {company} | Previous calls: {call_count}
"""
    if summaries:
        context += f"Recent context: {summaries}\n"

    context += """
INSTRUCTIONS:
- Greet by name: "Welcome back, [name]!"
- Confirm identity: "Am I speaking with [name]?"
- Skip full name/status collection if confirmed
- Still call store_caller_info to refresh the record
"""
    return context