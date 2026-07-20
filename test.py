from dotenv import load_dotenv
import os

load_dotenv()

print("MONGO_URI =", os.getenv("MONGO_URI"))