import torch
from peft import PeftModel, PeftConfig
from transformers import BertForSequenceClassification, AutoTokenizer

import numpy as np

def init_model():
    # Load peft config for pre-trained checkpoint etc.
    peft_model_id = "results"
    config = PeftConfig.from_pretrained(peft_model_id)

    # load base LLM model and tokenizer
    model = BertForSequenceClassification.from_pretrained(peft_model_id, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    model = model.merge_and_unload()

    model.eval()

    print("Peft model loaded")

    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = init_model()

    inputs = tokenizer.encode("This movie was really good", return_tensors="pt")#.to(device)
    print(inputs)
    res = model(inputs.clone().detach())
    print(res)
    predicted_label = np.argmax(res.logits.detach().numpy())
    print(predicted_label)
