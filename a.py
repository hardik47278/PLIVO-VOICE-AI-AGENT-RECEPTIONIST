import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb+srv://hardik:hardik123@cluster0.1ijo6.mongodb.net/workmates_voice?appName=Cluster0"
DB_NAME = "workmates_voice"


async def full_reset():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]

    collections_to_clear = [
        "caller_memory",
        "inbound_transcripts",
        "call_transcripts",
        "inbound_leads",
        "appointments",
    ]

    for col_name in collections_to_clear:
        result = await db[col_name].delete_many({})
        print(f"🗑️  {col_name}: deleted {result.deleted_count} documents")

    client.close()
    print("\n✅ Full reset done. Ready for fresh testing.")


async def delete_one_caller(phone_number: str):
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]

    result = await db["caller_memory"].delete_one({"phone_number": phone_number})
    print(f"🗑️  caller_memory: deleted {result.deleted_count} document(s) for {phone_number}")

    client.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # python reset_db.py 917991148912
        phone = sys.argv[1]
        asyncio.run(delete_one_caller(phone))
    else:
        # python reset_db.py
        asyncio.run(full_reset())