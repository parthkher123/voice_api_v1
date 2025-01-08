import os
import json
import base64
import asyncio
import websockets
import aiohttp
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # requires OpenAI Realtime API Access
PORT = int(os.getenv('PORT', 5050))

# System Message modified to include Indian tone and potential for Indian language responses
SYSTEM_MESSAGE = (
    "You are an AI assistant specializing in loan details collection, specifically for Indian clients. "
    "Your task is to interact with the client in a friendly, polite, and culturally appropriate Indian tone. "
    "You should carefully ask for their personal and loan details, such as their name, address, loan amount, loan purpose, "
    "and other relevant details. Use common Indian expressions and ensure that you respond in an approachable, respectful, "
    "and professional manner. If the client uses Hindi, Gujarati, or other Indian languages, you should respond in the same language."
)

VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

# Store the conversation in a text file
def store_conversation(client_name, conversation):
    """Store the conversation in a text file."""
    # Use the client's name for the filename or default to 'Unknown_Client'
    filename = f"{client_name.replace(' ', '_')}_loan_conversation.txt" if client_name != "Unknown Client" else "Unknown_Client_loan_conversation.txt"
    
    # Write the conversation to the text file
    with open(filename, "w") as file:
        file.write(conversation)
    print(f"Conversation stored in {filename}")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("Please wait while we connect your call to the AI voice assistant, powered by Twilio and OpenAI.")
    response.pause(length=1)
    response.say("OK, you can start talking!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    # Use aiohttp for WebSocket connection to OpenAI
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            await send_session_update(openai_ws)
            stream_sid = None
            client_name = "Unknown Client"
            conversation_log = ""

            async def receive_from_twilio():
                """Receive audio data from Twilio and send it to OpenAI Realtime API."""
                nonlocal stream_sid, conversation_log
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data['event'] == 'media' and openai_ws.closed == False:
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            await openai_ws.send_json(audio_append)
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            print(f"Incoming stream has started {stream_sid}")
                except WebSocketDisconnect:
                    print("Client disconnected.")
                    if openai_ws.closed == False:
                        await openai_ws.close()

            async def send_to_twilio():
                """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
                nonlocal stream_sid, client_name, conversation_log
                try:
                    async for openai_message in openai_ws:
                        response = openai_message.json()
                        if response['type'] in LOG_EVENT_TYPES:
                            print(f"Received event: {response['type']}", response)
                        if response['type'] == 'session.updated':
                            print("Session updated successfully:", response)
                        if response['type'] == 'response.audio.delta' and response.get('delta'):
                            # Audio from OpenAI
                            try:
                                audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(audio_delta)
                            except Exception as e:
                                print(f"Error processing audio data: {e}")
                        if response['type'] == 'response.content.done':
                            # Capture AI response and store it in the conversation log
                            ai_response = response.get('content', "")
                            print(f"AI says: {ai_response}")
                            conversation_log += f"AI: {ai_response}\n"
                            # If the AI asks for client name, we store it
                            if "name" in ai_response.lower() and client_name == "Unknown Client":
                                client_name = ai_response.split("name")[-1].strip()

                except Exception as e:
                    print(f"Error in send_to_twilio: {e}")

            await asyncio.gather(receive_from_twilio(), send_to_twilio())

            # Once the call ends, save the conversation
            store_conversation(client_name, conversation_log)

async def send_session_update(openai_ws):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send_json(session_update)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
