from openai import OpenAI
import gradio as gr
import warnings
import pyttsx3
import threading
from time import sleep
import cv2
from src.yolo_detector import YoloDetector
from src.text_extractor import TextExtractor
from src.face_recognizer import recognize_faces
from src.prompt import system
import os
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv
import subprocess
import sys
warnings.filterwarnings("ignore")

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("voice", "english-us")
load_dotenv()
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
extractor = TextExtractor(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

client = OpenAI(
    api_key=os.environ.get("api"),
)



messages = [{"role": "system", "content":system}]

def addusermessage(message):
    global messages
    messages.append({"role": "user", "content": message})

def addsystemmessage(message):
    global messages
    messages.append({"role": "system", "content": message})

def addassistantmessage(message):
    global messages
    messages.append({"role": "assistant", "content": message})

def gptreply():
    global messages
    response = client.chat.completions.create(messages=messages,model="gpt-3.5-turbo")
    ChatGPT_reply =response.choices[0].message.content
    return ChatGPT_reply


def CustomChatGPT(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    user_input=transcriber({"sampling_rate": sr, "raw": y})["text"]
    print(user_input)
    if 'read' in str(user_input).lower():
        print("reading")
        addusermessage(user_input)
        extracted_text = extractor.extract_text_from_image("frame.jpg")
        print(extracted_text)
        text='extracted text' + extracted_text
        addsystemmessage(text)
        ChatGPT_reply =gptreply()
        addassistantmessage(ChatGPT_reply)
        sleep(1)
    if "save this person as" in str(user_input).lower():
        words = str(user_input).split()
        last_word = words[-1]
        save_screenshots(last_word,'frame.jpg')
        run_detector_script()
        addusermessage(user_input)
        ChatGPT_reply = gptreply()
        addassistantmessage(ChatGPT_reply)
    else:
        addusermessage(user_input)
        ChatGPT_reply = gptreply()
        addassistantmessage(ChatGPT_reply)
    engine.save_to_file(ChatGPT_reply, "response.mp3")
    engine.runAndWait()
    return "response.mp3"

def object_detection():
    detector = YoloDetector()
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite('frame.jpg', frame)
        log_string = detector.detect_objects()
        if log_string:
            announcement = log_string
            print(announcement)
            print("detection working")
            if 'person' in announcement:
                faces = recognize_faces("frame.jpg")
                if len(faces)!= 0:
                    messages.append({"role": "system", "content": announcement +"persons detected:" + str(faces)})
                    print(faces)
                else:
                    print("no familiar faces")
            else:
                messages.append({"role": "system", "content": announcement})
        sleep(5)  
    cap.release()
    cv2.destroyAllWindows()

def save_screenshots(directory_name, image_file):
    parent_directory = "training"
    full_directory = os.path.join(parent_directory, directory_name)
    if not os.path.exists(full_directory):
        os.makedirs(full_directory)
    for i in range(15):
        file_name = os.path.join(full_directory, f"screenshot_{i+1}.jpg")
        try:
            with open(image_file, 'rb') as source, open(file_name, 'wb') as dest:
                dest.write(source.read())
            print(f"Screenshot {i+1} saved: {file_name}")
        except IOError as e:
            print(f"Error saving screenshot {i+1}: {e}")
        cv2.waitKey(500)

def run_detector_script():
    try:
        subprocess.run([sys.executable, "src/detector.py", "--train", "-m=hog"], check=True)
        print("Detector script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing detector script: {e}")

# Start object detection in a separate thread
object_detection_thread = threading.Thread(target=object_detection)
object_detection_thread.daemon = True
object_detection_thread.start()

title = "AVA - AI Vision Assistant"
description = '''AVA is your friendly vision assistant designed to help you navigate the world with confidence. Using real-time data from your surroundings, AVA can describe objects, people, and even read text aloud.
\n
Key Features:
\n
Real-time Object & People Detection: Stay informed about your surroundings.\n
Text Reading: Need something read? AVA can handle it with OCR.\n
Person Recognition: Get notified when familiar faces are nearby.\n
Reliable Information: AVA only speaks when it has accurate data.\n
Let AVA be your eyes and guide you through your day!
'''


demo = gr.Interface(fn=CustomChatGPT, live=True, title=title, description=description, inputs=gr.Audio(sources="microphone"), outputs="audio")
demo.launch(share=True)


