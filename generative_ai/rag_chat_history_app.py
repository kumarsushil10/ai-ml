import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



st.title("Document Analyser")

if 'store' not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
        st.session_state.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


uploaded_files = st.sidebar.file_uploader(label="Choose PDF.",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        tempPdf = f"./temp.pdf"
        with open(tempPdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    splitted_documents = textSplitter.split_documents(documents)
    embedding = OllamaEmbeddings(model="llama3.2")
    vector_db = FAISS.from_documents(splitted_documents,embedding)
    retriever = vector_db.as_retriever()

    system_prompt = (
        """ 
        You are an AI assistant.
        Your goal is to provide clear, accurate, and well-structured answers based on the given context.
        Carefully analyze the context before answering, and explain in a simple and understandable way.
        If necessary, break complex concepts into smaller parts to enhance comprehension.
        <context> {context} </context>
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)
    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Streamlit 
    question = st.chat_input("Ask question related to provide document")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question:
         # Human 
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Ai
        answer = conversational_rag_chain.invoke({"input":question},config={"configurable":{"session_id":"default01"}})["answer"]
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})    
else:
    st.write("Please Upload file(.pdf) to analyse")
    st.session_state.store.clear()
    st.session_state.messages.clear()