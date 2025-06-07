from app.model import load_model
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(index_path="vectorstore/"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(index_path,embeddings,allow_dangerous_deserialization=True)

def answer_query(query , vectorstore, model_pipeline):
    docs = vectorstore.similarity_search(query,k=3)
    context = "\n".join([docs.page_count for doc in docs])
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
 
    output = model_pipeline(prompt)[0]['generated_text']

  
    answer = output[len(prompt):]

    return answer.strip()