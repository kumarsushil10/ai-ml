import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")


path = r"C:\Users\kmrsu\OneDrive\Desktop\workSpace\tutorial\generative_ai\science_1007.pdf"
loader = PyPDFLoader(file_path=path)
documents = loader.load()
textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
splitted_documents = textSplitter.split_documents(documents)
embedding = OllamaEmbeddings(model="llama3.2")
vector_db = FAISS.from_documents(splitted_documents,embedding)
retriever = vector_db.as_retriever()


llm = OllamaLLM(model="llama3.2",temperature=0.5)

prompt = ChatPromptTemplate.from_messages(
    [
        ("ai", "You are an AI assistant that answers questions based on the given contexts. <context>{context}</context>"),
        ("human", "{input}"),
    ]
)

documents_chain = create_stuff_documents_chain(llm,prompt)
retrieval_chain = create_retrieval_chain(retriever,documents_chain)



st.title("Document Analyser")
question = st.text_input("Ask question related to provide document")

if question :
    answer = retrieval_chain.invoke({"input":question})['answer']
    st.write(answer)