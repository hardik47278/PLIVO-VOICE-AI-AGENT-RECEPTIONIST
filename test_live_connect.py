import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

TOOL_DECLARATIONS = [
    {
        "name": "query_company_data",
        "description": "Retrieve Workmates Core2Cloud company knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["workmates"],
                    "description": "The company knowledge topic."
                }
            },
            "required": ["topic"]
        }
    },
]

async def main():
    client = genai.Client(api_key=API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        temperature=0.4,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                silence_duration_ms=500
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text="You are a test assistant.")]
        ),
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    print(f"Connecting with model={model}")
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            print("CONNECTED OK")
            await session.send_realtime_input(text="Hello")
           
            async for response in session.receive():
                print("GOT RESPONSE:", response)
                break
    except Exception as e:
        print("CONNECT ERROR:", repr(e))

asyncio.run(main())