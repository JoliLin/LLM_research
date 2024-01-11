import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 

import numpy as np

def init_model():
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = "results"
    config = PeftConfig.from_pretrained(peft_model_id)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_model_id )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    model = model.merge_and_unload()
    model.eval()

    print("Peft model loaded")

    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = init_model()
    
    data = """
    Amanda: I like to read fantastic novel.\n
    Jerry: Oh, me too. The lord of Rings is my favorite one. 
    """

    inputs = tokenizer.encode(data, return_tensors="pt")#.to(device)
    print(inputs)
    outputs = model.generate(inputs) 
    print(outputs)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
