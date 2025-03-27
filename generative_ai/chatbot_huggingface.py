import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

prompt = ChatPromptTemplate.from_messages(
    [("ai","Provide the answer based on the question"),
     ("human","Question:{question}")]
)

# llm = HuggingFaceEndpoint(model="meta-llama/Llama-3.2-3B-Instruct")
# output_parser = StrOutputParser()
# chain = prompt|llm|output_parser



def generate_response(question, llm, temperature, max_token):
    llm = HuggingFaceEndpoint(model=llm,temperature=temperature,max_new_tokens=max_token)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer


st.title("langchain chatbot")

st.sidebar.title("Analysis..")
models = ["meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B","meta-llama/Llama-3.1-8B","meta-llama/Meta-Llama-3-8B","google/gemma-2b","google/gemma-7b"]
llm = st.sidebar.selectbox("Select model",models)
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_Token = st.sidebar.slider("Max Token", min_value=50,max_value=250,value=75)

question = st.text_input("What in your Mind!")

if question:
    answer = generate_response(question=question,llm=llm,temperature=temperature,max_token=max_Token)
    st.write(answer)