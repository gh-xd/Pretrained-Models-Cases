from datasets import load_dataset

raw_datasets = load_dataset("imdb")
print(raw_datasets)

train_set = raw_datasets['train']
train_X = train_set.data['text']
train_y = train_set.data['label']

from transformers import AutoTokenizer
from pprint import pprint
import time
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

start_time = time.time()

# Generate more features like 'attention_mask', 'input_ids' into original dataset using MAP
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# Try SAMLL Subset
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_train_dataset = tokenized_datasets["test"]


# Fine-Tuning in PyTorch with
from transformers import AutoModelForSequenceClassification

# pretraining head of BERT model be thrown away, replaced by classification head which is randomly initialized
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Define a Trainer, need to instantiate a Training Arguments
from transformers import TrainingArguments, Trainer

# set a directory is a must: test_trainer
training_args = TrainingArguments("test_trainer")

# instantiate a Trainer:
trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

# To fine tune, just call
trainer.train()

# To see metrics, we need define compute_metrics function that takes predictions and labels

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.evaluate()

end_time = time.time()

print('Prep, Train, Eval Time: %.2f'%(end_time - start_time))

#...