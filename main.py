from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from langchain import OpenAI
from streamlit_chat import message
import streamlit as st
import sys
import os
from dotenv import load_dotenv


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs
        )
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk("index.json")

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk("index.json")
    response = index.query(input_text, response_mode="compact")
    return response.response


def init():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(
        page_title="Ontario Tenant Rights Chatbot", page_icon=":cityscape:"
    )


def main():
    init()
    # TODO: better GUI ????
    # TODO: message history
    # TODO: handle greetings and misc stuff
    
    st.title("Ontario Tenant Rights Chatbot :cityscape:")
    st.write(
        "This chatbot is designed to help you understand your rights as a tenant in Ontario based on knowledge from the [Residential Tenancies Act, 2006](https://www.ontario.ca/laws/statute/06r17). It is not a substitute for legal advice. If you need legal advice, please contact a lawyer or paralegal. As this is a student-made bot, please double check with the sources provided in the bot's responses."
    )

    message(
        "Hi, I'm the Ontario Tenant Rights Chatbot. I can help you understand your rights as a tenant in Ontario. What is your question?"
    )
    user_input = st.chat_input("Ask your question here...")

    if user_input:
        message(user_input, is_user=True)
        message(chatbot(user_input))


if __name__ == "__main__":
    main()