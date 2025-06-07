from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def ingest_pdf(pdf_path,index_path="vectorstore/"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local(index_path)
    print(f"{len(chunks)}")


if __name__ == "__main__":
    ingest_pdf("data/example.pdf")