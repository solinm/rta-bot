from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import streamlit as st
import sys
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    index.save_to_disk('index.json')
    
    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

def main():
    st.title("Ontario Tenant Rights Chatbot")

    if not os.path.exists("index.json"):
        st.write("Constructing the index. Please wait...")
        construct_index("docs")
        st.write("Index constructed.")
    
    user_input = st.text_area("Enter your text", height=200)
    if st.button("Submit"):
        response = chatbot(user_input)
        st.text("Response:")
        st.write(response)

if __name__ == "__main__":
    main()
