# AVA - AI Vision Assistant

AVA is an innovative AI-driven solution designed to provide essential support to individuals with visual impairments. Leveraging cutting-edge deep learning technologies, AVA offers real-time object and people detection, text reading capabilities, person recognition, and interactive conversational features. This README provides an overview of the project, its features, setup instructions, and usage guidelines.

## Features

- **Real-time Object & People Detection**: Stay informed about your surroundings.
- **Text Reading**: Extract text from images and read aloud.
- **Person Recognition**: Identify familiar faces nearby.
- **Natural Language Interaction**: Engage in seamless conversation powered by GPT3.5 with AVA.
- **User-Friendly Interface**: Intuitive UI powered by Gradio for easy interaction.


## Project Folder Contents

- `Mainapp_AtoA.py`: The main Python script to run the AVA application. It offers Voice input to Voice output interaction.
- `appTtoT.py`: AVA application but Text to Text Interaction.
- `appAtoT.py`: AVA application but Audio to Text Interaction.
- `requirements.txt`: File listing all required Python dependencies.
- `yolo_detector.py`: Python script containing the YOLO v8 object detection implementation.
- `face_recognizer.py`: Python script containing the face recognition implementation.
- `text_extractor.py`: Python script containing the text extraction functionality using Tesseract OCR.
- `src/`: Directory containing system prompt for the chatbot.
- `output/`: Directory containing trained face recognizer model.
- `yolov8s.pt`: Pretrained YOLO v8 object detection Model.

## Setup Instructions

1. **Install Dependencies**: Ensure that Python 3.x is installed on your system. Install the required dependencies by running:

2. **Set Up Environment Variables**: Obtain API keys for OpenAI GPT-3.5 Turbo. Create a `.env` file in the project directory and add the API key.

3. **Install DroidCam**: Install the DroidCam application on both your laptop and mobile device from the respective app stores or from the official website (https://www.dev47apps.com/).

4. **Connect IP Camera**: Launch the DroidCam application on your mobile device and note the IP address and port number displayed. Open the DroidCam client on your laptop, enter the IP address and port number, and connect to the IP camera feed.

5. **Run the Application**: Launch the AVA application by running `Mainapp_AtoA.py`



## Usage

- Launch the AVA application and grant necessary permissions.
- Interact with AVA using voice commands .
- Ask AVA about objects, people, or text in your surroundings.
- Use keywords like "read" to request text reading from captured images.
- Enjoy seamless assistance and accessibility features provided by AVA.


