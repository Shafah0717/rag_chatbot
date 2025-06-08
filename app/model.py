from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv
import os



def load_model():
    load_dotenv() 
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    token = os.getenv("tokenkey")  

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
    torch_dtype=torch.float16, token=token)

    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
