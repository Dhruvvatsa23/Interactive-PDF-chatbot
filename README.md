# Interactive-PDF-chatbot

# Overview
Welcome to the Interactive PDF Chatbot, a smart, AI-driven application that lets you interact with PDF documents in a conversational manner. Whether you want to ask questions about the content of the PDF or just chat about the topics covered in it, this bot is designed to respond with informative, engaging answers. You can choose to interact with the chatbot using text or speech.

# Features
#### Text-to-Speech (TTS): The bot can read out its responses aloud, making it feel more conversational.
#### Speech-to-Text (STT): You can ask questions by voice. The bot will listen to your voice and transcribe it into text.
#### AI-Powered Responses: The chatbot is powered by advanced language models, making it capable of answering questions based on the content of the PDF.
#### PDF Processing: The bot automatically processes a PDF, splitting it into manageable chunks and storing its content for easy access.
#### Interactive Conversations: You can ask questions, and the bot will respond based on the content of the PDF. It can also suggest related topics if it doesn't find a direct answer.

# Installation
#### Setup ollama

- download and setup from: 'https://ollama.com/download'
- run the following command in the enivronment created:
      - ollama pull mistral:instruct
#### 1. To run this chatbot, you'll need to set up your environment by installing some dependencies.

      - git clone https://github.com/Dhruvvatsa23/Interactive-PDF-chatbot.git
      - cd Interactive-PDF-chatbot

#### 2. Install Python dependencies: Make sure you have Python installed (version 3.8 or higher), and install the required libraries by running:

      - pip install -r requirements.txt

# How to Use
#### 1. Launch the Chatbot: After installation, you can start the chatbot by running:

      - python chatbot.py

#### 2.Choose Input Mode: After starting the bot, you will be prompted to choose your preferred input mode:
- Type text to interact using text.
- Type speech to interact using your voice.

#### 3. Ask Questions:
- If you're using text mode, simply type your question.
- If you're using speech mode, press and hold the spacebar to record your question. Release the spacebar when you're done speaking.

#### 4. Get Responses: 
- The chatbot will provide answers based on the content of the PDF you provided.
- If you chose speech mode, it will speak the answers out loud; otherwise, it will display the answers on the screen.

#### 5. End the Chat: 
- To exit the chatbot, type "exit" or "quit". Your conversation will be saved as a log file.

# Example
Here’s a simple walkthrough of how a conversation might go:

#### Welcome Message: 
- The chatbot greets you and asks for your preferred input mode (text or speech).

#### You Ask a Question:
- You could ask: "What is karma?" if the PDF contains that information.
- The chatbot responds: "Karma is a concept discussed in the PDF. It refers to..."
- End the Chat: When you're done, just type or say "exit", and the chatbot will end the session, saving the chat log.

# PDF Processing
The bot uses AI to split the PDF into sections and store them for quick access. This allows the bot to retrieve answers from different parts of the document efficiently.

# Dependencies
- Whisper (for Speech-to-Text): Transcribes voice input into text.
- pyttsx3 (for Text-to-Speech): Converts text responses into spoken words.
- ChromaDB: Stores processed PDF content and retrieves relevant information during chats.
- Langchain Community Models: Powers the chatbot's responses, ensuring they are context-aware and conversational.
- Sounddevice: Used to capture voice input.

# Future Improvements
- Support for More Languages: Currently, the chatbot supports English. In future versions, we plan to add support for other languages.
- Improved Response Times: Optimizing the system to handle larger PDFs faster and more efficiently.



