import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompts = ChatPromptTemplate.from_messages(
    [
        ("ai", "Provide accurate answer. based on Question"),
        ("human","Question:{question}")
    ]
)

llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()
chain = prompts|llm|output_parser




st.title("Langchain and Ollama demo")
input_text = st.text_input("Enter your Question: ")

if input_text:
    st.write(chain.invoke({"question":input_text}))

