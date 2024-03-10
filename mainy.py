import openai
import gradio
import json
import warnings

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




demo = gradio.Interface(
    fn=CustomChatGPT, inputs = "text", outputs = "text", title = "AVA"
    )

demo.launch(share=True)