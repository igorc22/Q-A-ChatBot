import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()


groq_api_key = os.getenv("CROQ_API_KEY")
os.environ['GOOGLE_API_KEY']= os.getenv("GOOGLE_API_KEY")

st.title("Model Documents Q&A")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")


prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}

"""
)


def embedding():
    if "vectors" not in st.session_state: 

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001") ## Setup of the embedding technique
        st.session_state.loader=PyPDFDirectoryLoader("./pdf") ## Reading the pdf from the directory
        st.session_state.docs=st.session_state.loader.load() ## Document Load
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=100000,chunk_overlap=200) ## Chunk Creation, chunk_overlap means that in the end/begin of two chunks 200 characters can overlap
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) ## splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## Vector Database


prompttext = st.text_input("Enter Your Question from the Documents")

if st.button("Documents Embedding"):
    embedding()
    st.write("Vector Store DB is Ready")


if prompttext: ## If user write something in prompttext then this condition is true
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompttext})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")