from datasets import load_dataset 
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# huggingface hub model id
model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# prepare dataset
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(
    range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(
    range(500))

# load model from the hub
model = BertForSequenceClassification.from_pretrained(model_id, num_labels=2)

# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# Define LoRA Config
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                         r=1,
                         lora_alpha=1,
                         lora_dropout=0.1)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100

# Define training args
training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=10,
                                  per_device_train_batch_size=8)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# Save our LoRA model & tokenizer results
peft_model_id = "results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)
