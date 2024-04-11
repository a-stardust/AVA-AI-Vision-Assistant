from openai import OpenAI
import gradio as gr
import warnings
import threading
from time import sleep
import keyboard
import cv2
from yolo_detector import YoloDetector
from text_extractor import TextExtractor
from face_recognizer import recognize_faces
from src.prompt import system
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

extractor = TextExtractor(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

warnings.filterwarnings("ignore")
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("api"),
)


messages = [{"role": "system", "content":system}]



def CustomChatGPT(audio):

    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    user_input=transcriber({"sampling_rate": sr, "raw": y})["text"]

    if 'read' in user_input.lower():
        print("reading")
        messages.append({"role": "user", "content": user_input})
        extracted_text = extractor.extract_text_from_image("frame.jpg")
        print(extracted_text)
        messages.append({"role": "system", "content": f'extracted text:{extracted_text}'})

        response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo")
        
        ChatGPT_reply =response.choices[0].message.content
        messages.append({"role": "assistant", "content": ChatGPT_reply})
    if "save the person as" in str(user_input).lower():
        words = str(user_input).split()
        last_word = words[-1]
        save_screenshots(last_word,'frame.jpg')
        run_detector_script()
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo")
        ChatGPT_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ChatGPT_reply})
    else:
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo")
        ChatGPT_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ChatGPT_reply})

    return ChatGPT_reply

def object_detection():
    detector = YoloDetector()
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        cv2.imwrite('frame.jpg', frame)
        log_string = detector.detect_objects()
        if log_string:
            # Announce detected objects
            announcement = log_string
            print(announcement)
            print("detection working")
            if 'person' in announcement:
                faces = recognize_faces("frame.jpg")
                messages.append({"role": "system", "content": announcement +"persons detected:" + str(faces)})
                print(faces)
            # Append announcement to conversation
            else:
                messages.append({"role": "system", "content": announcement})

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
            break

        sleep(5)  # Wait for 5 seconds before checking again
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
        # Try to run the script using the same Python interpreter as the main application
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
demo = gr.Interface(fn=CustomChatGPT, live=True, inputs=gr.Audio(sources="microphone"), outputs="text",title=title, description=description)
demo.launch(share=True)


