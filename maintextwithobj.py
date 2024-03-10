import openai
import gradio
import json
import warnings
import json
import threading
from time import sleep
from yolo_detector import YoloDetector

warnings.filterwarnings("ignore")
with open('GPT_SECRET_KEY.json') as f:
    data = json.load(f)

openai.api_key = data["API_KEY"]
print(data["API_KEY"])

messages = [{"role": "system", "content": "You are AVA , visual assistant to blind people. u will be integrated with a camera and a object detection model which sends u the objects, people and data regarding the surrounding in real time. u have to talk to the user about what is infront of them and answer their queries using the data which is streamed to u every 5 seconds. talk less"}]



def CustomChatGPT(user_input):
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
    while True:
        log_string = detector.detect_objects()
        if log_string:
            # Announce detected objects
            announcement = f"Objects detected: {log_string}"
            print(announcement)
            # Append announcement to conversation
            messages.append({"role": "system", "content": announcement})
        sleep(5)  # Wait for 5 seconds before checking again


# Start object detection in a separate thread
object_detection_thread = threading.Thread(target=object_detection)
object_detection_thread.daemon = True
object_detection_thread.start()

demo = gradio.Interface(
    fn=CustomChatGPT, inputs = "text", outputs = "text", title = "AVA"
    )

demo.launch(share=True)


