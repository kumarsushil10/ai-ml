import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [("ai","Provide accurate answer in 50 to 100 words"),
     ("human","Question:{question}")]
     )
llm = ChatGroq(model="llama-3.2-11b-vision-preview",)
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

st.title("Groq Demo..")
question = st.text_input("Give me Question ")

if question:
    st.write(chain.invoke({"question":question}))
