import gradio as gr
import openai
import pyttsx3
import json
import threading
from time import sleep
from yolo_detector import YoloDetector

# Load OpenAI API key from JSON file
with open('GPT_SECRET_KEY.json') as f:
    data = json.load(f)

openai.api_key = data["API_KEY"]

# Global variable to hold the chat history, initialised with system role
conversation = [
    {"role": "system", "content": "You are AVA, you provide vision assistance to blind people. You will be integrated with a camera and an object detection model which sends you the objects, people, and data regarding the surrounding in real-time. You have to talk to the user about what is in front of them and answer their queries using the data which is streamed to you every 5 seconds. Talk less."}
]

def transcribe(audio):
    # Whisper API for transcription
    transcript = openai.Transcribe.create("whisper-1", audio)

    # Append user's input to conversation
    conversation.append({"role": "user", "content": transcript["text"]})

    # ChatGPT API for generating response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # System_message is the response from ChatGPT API
    system_message = response["choices"][0]["message"]["content"]

    # Append ChatGPT response (assistant role) back to conversation
    conversation.append({"role": "assistant", "content": system_message})

    # Text to speech conversion
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("voice", "english-us")
    engine.say(system_message)
    engine.runAndWait()

    return system_message

def object_detection():
    detector = YoloDetector()
    while True:
        log_string = detector.detect_objects()
        if log_string:
            # Announce detected objects
            announcement = f"Objects detected: {log_string}"
            print(announcement)

            # Append announcement to conversation
            conversation.append({"role": "system", "content": announcement})

            # Text to speech conversion for announcing detected objects
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("voice", "english-us")
            engine.say(announcement)
            engine.runAndWait()

        sleep(5)  # Wait for 5 seconds before checking again

# Gradio interface for audio input
bot = gr.Interface(fn=transcribe, inputs=gr.Audio(sources="microphone", type="filepath"), outputs="audio")

# Start object detection in a separate thread
object_detection_thread = threading.Thread(target=object_detection)
object_detection_thread.daemon = True
object_detection_thread.start()

# Launch the chatbot interface
bot.launch(share=True)
