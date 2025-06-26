import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

def train_model():
    # Load data
    df = pd.read_csv('processed_recipes.csv')
    dataset = Dataset.from_pandas(df)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on device: {device}")

    # Tokenize data
    def tokenize_function(examples):
        encodings = tokenizer(examples['formatted'], padding='max_length', truncation=True, max_length=256)
        encodings['labels'] = encodings['input_ids'].copy()  # Set labels for language modeling
        return encodings

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['formatted'])
    tokenized_dataset.set_format('torch')

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Shortened for speed
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        eval_strategy='no'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    model.save_pretrained('recipe_model')
    tokenizer.save_pretrained('recipe_model')

if __name__ == "__main__":
    train_model()