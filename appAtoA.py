from openai import OpenAI
from playsound import playsound
import gradio as gr
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
import numpy as np

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("voice", "english-us")

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
    print (user_input)

    if 'read' in str(user_input):
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

    else:
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo")
        ChatGPT_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ChatGPT_reply})

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
                # names=''
                # if len(faces) > 1:
                #         for i in range(0,len(faces)-1):
                #             names+=faces[i]
                #             names+=' and '
                #         names+=faces[len(faces)-1]
                #         names= 'say hi to' + names
                #         engine.save_to_file(names, "person.mp3")
                #         sleep(1)                    #LINE ADDED
                #         engine.runAndWait()
                #         playsound('person.mp3')                    

                # elif len(faces)==1 :
                #     names='say hi to' + str(faces[0])
                #     engine.save_to_file(names, "person.mp3")
                #     engine.runAndWait()
                #     playsound('/person.mp3')                    
            # Append announcement to conversation

            else:
                messages.append({"role": "system", "content": announcement})

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
            break

        sleep(5)  # Wait for 5 seconds before checking again
    cap.release()
    cv2.destroyAllWindows()

# Start object detection in a separate thread
object_detection_thread = threading.Thread(target=object_detection)
object_detection_thread.daemon = True
object_detection_thread.start()


demo = gr.Interface(fn=CustomChatGPT, inputs=gr.Audio(sources="microphone"), outputs="audio")



demo.launch(share=True)


