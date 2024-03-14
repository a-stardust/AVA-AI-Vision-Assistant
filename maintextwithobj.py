import openai
import gradio
import json
import warnings
import json
import threading
from time import sleep
import keyboard
import cv2
from yolo_detector import YoloDetector
from text_extractor import TextExtractor
from face_recognizer import recognize_faces

from src.prompt import system

extractor = TextExtractor(r'C:\Program Files\Tesseract-OCR\tesseract.exe')



warnings.filterwarnings("ignore")
with open('GPT_SECRET_KEY.json') as f:
    data = json.load(f)

openai.api_key = data["API_KEY"]
print(data["API_KEY"])

messages = [{"role": "system", "content":system}]



def CustomChatGPT(user_input):
    if 'read' in user_input:
        print("reading")
        extracted_text = extractor.extract_text_from_image("frame.jpg")
        print(extracted_text)
        messages.append({"role": "system", "content": extracted_text})
        response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages)
        ChatGPT_reply = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": ChatGPT_reply})

    else:
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages
        )
        ChatGPT_reply = response["choices"][0]["message"]["content"]
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
            announcement = f"Objects detected: {log_string}"
            print(announcement)
            print("detection working    ")
            if 'person' in announcement:
                faces = recognize_faces("frame.jpg")
                messages.append({"role": "system", "content": 'persons detected:' + str(faces) + 'objected detected:'+ announcement})
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

# Start object detection in a separate thread
object_detection_thread = threading.Thread(target=object_detection)
object_detection_thread.daemon = True
object_detection_thread.start()

demo = gradio.Interface(
    fn=CustomChatGPT, inputs = "text", outputs = "text", title = "AVA"
    )

demo.launch(share=True)


