from twilio.twiml.voice_response import VoiceResponse
from flask import Flask, request, Response
from .speech_to_text import transcribe_audio
from .text_to_speech import synthesize_speech
from general_bot.agent.phone_bot.phone_generative_agent import GenerativeAgent

app = Flask(__name__)
agent = GenerativeAgent()

@app.route("/start_call", methods=["POST"])
def start_call():
    # Initial response to start the call
    response = VoiceResponse()
    response.say("Hello, this is your AI assistant. How can I help you today?")
    response.record(max_length=60, action="/process_audio")
    return Response(str(response), mimetype="text/xml")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    # Get recorded audio and convert to text
    audio_url = request.form['RecordingUrl']
    recognized_text = transcribe_audio(audio_url)
    
    # Process AI reasoning and generate response
    next_response = agent.reason_and_generate_next_action(recognized_text)
    
    # Convert response text to speech
    audio_response = synthesize_speech(next_response)
    
    # Respond back in the call
    response = VoiceResponse()
    response.play(audio_response)  # Playing generated speech back to the user
    response.record(max_length=60, action="/process_audio")  # Continue conversation
    return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    app.run(port=5000)
