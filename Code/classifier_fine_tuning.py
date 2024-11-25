import wandb
wandb.login(key="Your Key")
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
from typing import Dict, List
import json
import gc
from transformers import GenerationConfig
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
HF_token ="Your HF token"
login(token = HF_token)
generation_config=GenerationConfig.from_pretrained("openai-community/gpt2")
generation_config.repetition_penalty=0.01
CLASSIFICATIONS = ["KM", "WKHM", "KC", "CKM", "KHM"]
class RouterClassificationDataset:
    def __init__(self, prompts: List[str], classifications: List[str], tokenizer):
        self.prompts = prompts
        self.classifications = classifications
        self.tokenizer = tokenizer
        
        # Validate classifications
#         for cls in self.classifications:
#             if cls not in CLASSIFICATIONS:
#                 raise ValueError(f"Invalid classification: {cls}. Must be one of {CLASSIFICATIONS}")
    
    def format_data(self) -> Dict:
        # Format each example as: "Query: {prompt}\nClassification: <{classification}>"
        formatted_texts = [
            f"Query: {prompt}\nClassification: <{cls}>"
            for prompt, cls in zip(self.prompts, self.classifications)
        ]
        
        # Tokenize the texts
        encodings = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone()
        }
    
    def to_dataset(self) -> Dataset:
        formatted_data = self.format_data()
        return Dataset.from_dict(formatted_data)

def prepare_model_and_tokenizer():
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8b-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8b-Instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer
#IMPORT YOUR OWN DATASET HERE
sample_prompts = prompts.copy()

sample_classifications = CLASSIFICATIONS.copy()
model, tokenizer = prepare_model_and_tokenizer()

dataset = RouterClassificationDataset(sample_prompts, sample_classifications, tokenizer)
full_dataset = dataset.to_dataset()

# Split dataset
train_test_split = full_dataset.train_test_split(test_size=0.2)
def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model,
    tokenizer,
    output_dir: str = "./router_classification_model"
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=30,
        save_steps=30,
        warmup_steps=30,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model()
train_model(
    train_dataset=train_test_split["train"],
    val_dataset=train_test_split["test"],
    model=model,
    tokenizer=tokenizer
)
def predict(prompt: str, model, tokenizer, max_length: int = 100):
    """
    Generate classification prediction for a single prompt
    """
    input_text = f"Query: {prompt}\nClassification: "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    if(max_length=="auto"):
        max_length=inputs.input_ids.shape[1]+10
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.1,  # Low temperature for more focused predictions
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract classification from between < and >
    import re
    print("Generated Response:"+ prediction)
    match = re.search(r'<([^>]+)>', prediction)
    if match:
        return match.group(1)
    return None
