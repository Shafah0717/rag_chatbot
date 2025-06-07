from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
def load_model():
    model_name = "tiiuaetiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
