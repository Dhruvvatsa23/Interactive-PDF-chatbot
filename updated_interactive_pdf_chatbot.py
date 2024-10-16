import asyncio
import aiohttp
import chromadb
import sounddevice as sd
import whisper
import os
import numpy as np
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
import pyttsx3
from scipy.io.wavfile import write
import keyboard
import time
import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Initialize Whisper model and ChromaDB
whisper_model = whisper.load_model("base")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate

# Global variable for vectorstore
vectorstore = None

def text_to_speech(text, input_mode):
    if input_mode == "speech":
        try:
            start_time = time.time()
            engine.say(text)
            engine.runAndWait()
            end_time = time.time()
            print(f"TTS time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"An error occurred during text-to-speech: {e}")
    else:
        print("\nChatbot:", text)

# def get_user_input(input_mode):
#     if input_mode == "text":
#         return input("Enter your response (or type 'exit' to end the chat): ")
#     else:
#         print("Press and hold spacebar to start recording...")
#         while not keyboard.is_pressed('space'):
#             time.sleep(0.1)
#         return record_and_transcribe()

def get_user_input(input_mode):
    if input_mode == "text":
        return input("Enter your response (or type 'exit' to end the chat): ")
    else:
        print("Press and hold spacebar to start recording...")
        while not keyboard.is_pressed('space'):
            time.sleep(0.1)
        transcription = record_and_transcribe()
        # print(f"Debug - Transcribed input: '{transcription}'")  # Debug print
        return transcription

def record_and_transcribe():
    print("Recording started... Release spacebar to stop.")
    
    fs = 44100
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while keyboard.is_pressed('space'):
            time.sleep(0.1)

    print("Recording finished.")
    audio_data = np.concatenate(recording, axis=0)
    filename = "input_audio.wav"
    write(filename, fs, audio_data)
    
    print("Transcribing...")
    start_time = time.time()
    result = whisper_model.transcribe(filename)
    transcription = result["text"]                                                                                                                                                                                                                                                                                    
    end_time = time.time()
    print(f"STT time: {end_time - start_time:.2f} seconds")
    print(f"Transcription: {transcription}")
    return transcription

def welcome_message():
    welcome_text = "Welcome to the Interactive PDF Chatbot! Please choose your input mode: text or speech."
    print(welcome_text)
    # We'll use 'speech' mode for the welcome message to ensure it's spoken
    text_to_speech(welcome_text, "speech")
    while True:
        mode = input("Enter 'text' or 'speech': ").lower()
        if mode in ["text", "speech"]:
            return mode
        print("Invalid input. Please enter 'text' or 'speech'.")

def process_pdf(pdf_path):
    start_time = time.time()
    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
    )
    docs_after_split = text_splitter.split_documents(docs)
    
    # Create embeddings and store them in Chroma
    embedding = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs_after_split, collection_name="my_collection", embedding=embedding)
    collection.upsert(documents=[doc.page_content for doc in docs_after_split], ids=[f"id{i}" for i in range(len(docs_after_split))])
    end_time = time.time()
    print(f"PDF processing time: {end_time - start_time:.2f} seconds")
    return vectorstore, [doc.page_content for doc in docs_after_split]

def get_random_topic(pdf_content):
    return random.choice(pdf_content).split('.')[0]

@lru_cache(maxsize=100)
def cached_retrieval(question):
    return vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)

def activate_chatbot(pdf_path):
    global vectorstore
    print("Processing PDF...")
    vectorstore, pdf_content = process_pdf(pdf_path)
    
    input_mode = welcome_message()
    
    llm = ChatOllama(model="mistral:instruct", temperature=0.2, gpu_layers=50)
    integrated_prompt = PromptTemplate(
    template="""Based on the following context, answer the question. 
    If the question is not directly related to the context, "do not" attempt to draw any connections/relation between the content and the question. 
    Instead, clearly state that the question is not relevant to the content and take up any random topic from the context then suggest/ask to discuss over it. 
    Provide a natural, conversational response that both answers the question and encourages further discussion.

    Context: {context}
    
    Question: {question}
    
    Response: [If the question is relevant, provide a detailed, engaging answer. If not relevant, state that it's not related to the current content.]""",
    input_variables=["context", "question"])
    
    chat_with_pdf(vectorstore, llm, integrated_prompt, input_mode, pdf_content)

# def chat_with_pdf(vectorstore, llm, integrated_prompt, input_mode, pdf_content):
#     log = []
#     start_message = "Chatbot is active. " + ("Press spacebar to start recording your question." if input_mode == "speech" else "Type your question (or type 'exit' to end the chat).")
#     print(start_message)
#     text_to_speech(start_message, input_mode)
    
#     while True:
#         total_start_time = time.time()
#         user_question = get_user_input(input_mode)
        
#         if user_question.lower() in ["exit", "quit", "i want to quit"]:
#             exit_message = "Thank you for using the Interactive PDF Chatbot. Goodbye!"
#             text_to_speech(exit_message, input_mode)
#             break
        
#         llm_start_time = time.time()
#         retrieved_docs = cached_retrieval(user_question)
        
#         if retrieved_docs:
#             context = "\n".join([doc.page_content for doc in retrieved_docs])
#             response = llm.invoke(integrated_prompt.format(context=context, question=user_question))
#             llm_end_time = time.time()
#             print(f"LLM response time: {llm_end_time - llm_start_time:.2f} seconds")
            
#             # Print the LLM response regardless of the input mode
#             print("\nChatbot:", response.content)
            
#             text_to_speech(response.content, input_mode)
            
#             log.append({"question": user_question, "answer": response.content})
#         else:
#             random_topic = get_random_topic(pdf_content)
#             response = f"I'm sorry, but your question doesn't seem to be related to the content we're focusing on. Would you like to discuss {random_topic} instead? It's an interesting topic from the document we're exploring."
            
#             # Print the response for the case when no relevant information is found
#             print("\nChatbot:", response)
            
#             text_to_speech(response, input_mode)
        
#         total_end_time = time.time()
#         total_time = total_end_time - total_start_time
#         print(f"Total response time: {total_time:.2f} seconds")
#         if total_time > 3:
#             print("Warning: Response time exceeded 3 seconds.")

#     chat_log_filename = f"chat_log_{int(time.time())}.json"
#     with open(chat_log_filename, 'w') as f:
#         json.dump(log, f, indent=4)
#     print(f"Conversation saved in {chat_log_filename}")
def chat_with_pdf(vectorstore, llm, integrated_prompt, input_mode, pdf_content):
    log = []
    start_message = "Chatbot is active. " + ("Press spacebar to start recording your question." if input_mode == "speech" else "Type your question (or type 'exit' to end the chat).")
    print(start_message)
    text_to_speech(start_message, input_mode)
    
    while True:
        total_start_time = time.time()
        user_question = get_user_input(input_mode)
        
        # Convert user_question to lowercase for case-insensitive comparison
        user_question_lower = user_question.lower().strip()
        # print(f"Debug - Processed input: '{user_question_lower}'")  # Debug print
        
        if any(exit_phrase in user_question_lower for exit_phrase in ["exit", "quit", "i want to quit"]):
            exit_message = "Thank you for using the Interactive PDF Chatbot. Goodbye!"
            print("\nChatbot:", exit_message)
            text_to_speech(exit_message, input_mode)
            # print("Debug - Exit condition met. Ending chat.")  # Debug print
            break
        
        llm_start_time = time.time()
        retrieved_docs = cached_retrieval(user_question)
        
        if retrieved_docs:
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            response = llm.invoke(integrated_prompt.format(context=context, question=user_question))
            llm_end_time = time.time()
            print(f"LLM response time: {llm_end_time - llm_start_time:.2f} seconds")
            
            print("\nChatbot:", response.content)
            
            text_to_speech(response.content, input_mode)
            
            log.append({"question": user_question, "answer": response.content})
        else:
            random_topic = get_random_topic(pdf_content)
            response = f"I'm sorry, but your question doesn't seem to be related to the content we're focusing on. Would you like to discuss {random_topic} instead? It's an interesting topic from the document we're exploring."
            
            print("\nChatbot:", response)
            
            text_to_speech(response, input_mode)
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Total response time: {total_time:.2f} seconds")
        if total_time > 3:
            print("Warning: Response time exceeded 3 seconds.")

    chat_log_filename = f"chat_log_{int(time.time())}.json"
    with open(chat_log_filename, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Conversation saved in {chat_log_filename}")

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\dhruv\OneDrive\Desktop\brilworks_AI_practical\rag_llm\Karma - Sadhguru.pdf"  # Path to your local PDF
    activate_chatbot(pdf_path)
