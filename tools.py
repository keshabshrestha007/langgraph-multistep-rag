import os
import time
from typing import Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec,Pinecone
from langchain.schema import Document
import streamlit as st
"""
# loading environment variables
load_dotenv()

if os.getenv("PINECONE_API_KEY") is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
"""

if st.secrets["PINECONE_API_KEY"] is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")
else:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]


def retriever_tool(file: Optional[str] = None, file_name: Optional[str] = None):
    """
    Creates or retrieves a Pinecone-based retriever for the given file.
    If the vector index for the file does not exist, it creates one by loading
    and processing the file.
    Args:
        file (Optional[str]): Path to the file to be used for creating/retrieving the index.
        file_name (Optional[str]): Name of the file to be used for naming the index.
    Returns:
        A retriever object for querying the vector index.
    """
    if file_name:
        index_name = (file_name.replace("_","-")).lower()
    else:
        #index_name = os.getenv("PINECONE_INDEX_NAME", "langgraph-index")
        index_name = st.secrets.get("PINECONE_INDEX_NAME", "langgraph-index")

    
    pc = Pinecone(api_key= pinecone_api_key)
    
    # Ensure index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

       
    # Wait until index is ready
    while True:
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            break
        time.sleep(2)
    
    vectorstore = PineconeVectorStore(index_name=index_name,embedding=HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-V2"))
    
    

    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    # Check if vector index is empty
    is_empty = sum(stats["total_vector_count"] for _ in stats["namespaces"].values()) == 0


    if is_empty:
        documents = []
            
        print(f"Loading file: {file} for index: {index_name}")

        if file.endswith(".pdf"):
            try:
                # instantiate PDF loader
                loader = PyPDFLoader(file)
                docs = loader.load()
                for doc in docs:
                    if doc.page_content.strip():
                        doc.metadata["source"] = file
                        documents.append(doc)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        elif file.endswith(".txt"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        doc = Document(page_content=content, metadata={"source": file})
                        if doc.page_content.strip():
                            doc.metadata["source"] = file
                            documents.append(doc)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        elif file.endswith(".docx"):
            try:
                loader = Docx2txtLoader(file)
                docs = loader.load()
                for doc in docs:
                    if doc.page_content.strip():
                        doc.metadata["source"] = file
                        documents.append(doc)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        else:
            print(f"Unsupported file type: {file}; returning empty retriever")
            
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2"),
            index_name=index_name
        )

    
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
      