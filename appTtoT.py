from openai import OpenAI
from playsound import playsound
import gradio
import warnings
import pyttsx3
import threading
from time import sleep
import keyboard
import cv2
from yolo_detector import YoloDetector
from text_extractor import TextExtractor
from face_recognizer import recognize_faces
from src.prompt import system
import os
from dotenv import load_dotenv
import datetime
load_dotenv()

import os
from dotenv import load_dotenv
load_dotenv()


extractor = TextExtractor(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("voice", "english-us")

warnings.filterwarnings("ignore")
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

def CustomChatGPT(user_input):
    if 'read' in user_input.lower():
        print("reading")
        addusermessage(user_input)
        extracted_text = extractor.extract_text_from_image("frame.jpg")
        print(extracted_text)
        text='extracted text' + extracted_text
        addsystemmessage(text)
        ChatGPT_reply =gptreply()
        addassistantmessage(ChatGPT_reply)
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
    return ChatGPT_reply

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
            # Announce detected objects
            announcement = log_string
            print(announcement)
            print("detection working")
            if 'person' in announcement:
                faces = recognize_faces("frame.jpg")
                if len(faces)==0:
                    print("no person detected")
                    messages.append({"role": "system", "content": announcement})
                else:
                    messages.append({"role": "system", "content": announcement +"persons detected:" + str(faces)})
                    print(faces)
            else:
                messages.append({"role": "system", "content": announcement})

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
            break

        sleep(3)  # Wait for 5 seconds before checking again
    cap.release()
    cv2.destroyAllWindows()



def save_screenshots(directory_name, image_file):
    # Set the parent directory
    parent_directory = "training"

    # Create the full directory path
    full_directory = os.path.join(parent_directory, directory_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(full_directory):
        os.makedirs(full_directory)

    # Save 15 copies of the image file in the directory
    for i in range(15):
        file_name = os.path.join(full_directory, f"screenshot_{i+1}.jpg")
        try:
            # Copy the image file to the new location
            with open(image_file, 'rb') as source, open(file_name, 'wb') as dest:
                dest.write(source.read())
            print(f"Screenshot {i+1} saved: {file_name}")
        except IOError as e:
            print(f"Error saving screenshot {i+1}: {e}")
        cv2.waitKey(500)

import subprocess
import sys

def run_detector_script():
    try:
        subprocess.run([sys.executable, "detector.py", "--train", "-m=hog"], check=True)
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

demo = gradio.Interface(
    fn=CustomChatGPT, inputs = "text", outputs = "text",title=title, description=description
    )

demo.launch(share=True)


