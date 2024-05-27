import streamlit as st
import os
import time

from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv()

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def vector_embedding():
    '''
    This function is used to embed the documents using OpenAI Embeddings
    '''
    if "vectors" not in st.session_state:
        st.write("Embedding Documents. Please wait.....")
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs) ## Splliting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings) ## Vector OpenAI Embedding


st.title("LLama3 Local Demo")

llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the 
    question.
    <context>
    {context}
    </context>
    Question: {input}"""
)

input_prompt = st.text_input("Enter your question regarding the docs:")

# Button to embed documents
if st.button("Embed Documents"):
    vector_embedding()
    st.write("Vector Store DB is Ready")


if input_prompt:
    start_time = time.process_time()
   
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain) 
    
    response = retrieval_chain.invoke({"input": input_prompt})
    print("Response time:", time.process_time() - start_time)
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------------------")