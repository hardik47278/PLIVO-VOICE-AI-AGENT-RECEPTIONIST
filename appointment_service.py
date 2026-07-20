# appointment_service.py
from datetime import datetime
import pytz
import dateparser
from motor.motor_asyncio import AsyncIOMotorCollection

def get_ist_now():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def format_ist_date(date):
    ist = pytz.timezone("Asia/Kolkata")
    if date.tzinfo is None:
        date = ist.localize(date)
    return date.strftime("%A, %d %B %Y")

def get_date_string(date):
    return date.strftime("%Y-%m-%d")

async def store_caller_info(name, company, phone_number, request_uuid, inbound_lead_collection):
    if not name or not phone_number or not request_uuid:
        return "Missing required information."
    try:
        existing = await inbound_lead_collection.find_one({"requestUuid": request_uuid})
        if existing:
            print(f"⚠️ Lead already exists for session {request_uuid}")
            return "Success"

        await inbound_lead_collection.insert_one({
            "fullName": name.strip(),
            "phone": phone_number,
            "company": company.strip() if company else "Not Provided",
            "requestUuid": request_uuid,
            "callDate": datetime.utcnow()
        })
        print(f"✅ Inbound lead stored: {name} from {company or 'Not Provided'}")
        return "Success"
    except Exception as e:
        print(f"❌ Error storing lead: {e}")
        return "Noted"

async def book_appointment(name, date, purpose, phone_number, request_uuid, appointments_collection):
    if not name or not date or not purpose:
        return "Missing required appointment details."

    now_ist = get_ist_now()
    parsed_date = dateparser.parse(date, settings={
        "PREFER_DATES_FROM": "future",
        "TIMEZONE": "Asia/Kolkata",
        "RETURN_AS_TIMEZONE_AWARE": True
    })

    if not parsed_date:
        return "I couldn't understand the date. Please say something like 'tomorrow' or '26 October'."

    parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
    today = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)

    if parsed_date <= today:
        return "Appointment date must be in the future."

    date_string = get_date_string(parsed_date)

    try:
        await appointments_collection.insert_one({
            "name": name.strip(),
            "date": date_string,
            "purpose": purpose.strip(),
            "phoneNumber": phone_number or "Unknown",
            "requestUuid": request_uuid or "unknown",
            "source": "voice_call_plivo",
            "timezone": "IST",
            "status": "pending",
            "createdAt": datetime.utcnow(),
            "storedAtIST": now_ist.isoformat()
        })
        formatted = format_ist_date(parsed_date)
        return f"Appointment successfully booked for {name}! Date: {formatted}, Purpose: {purpose}."
    except Exception as e:
        return f"Issue storing appointment: {str(e)}"