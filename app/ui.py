import gradio as gr
from app.rag_chain import load_vectorstore, answer_query
from app.model import load_model

vectorstore = load_vectorstore()
model_pipeline = load_model()

def chat(query):
    return answer_query(query, vectorstore, model_pipeline)

gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="📚 RAG Chatbot",
    description="Ask anything based on the uploaded PDF"
).launch()
