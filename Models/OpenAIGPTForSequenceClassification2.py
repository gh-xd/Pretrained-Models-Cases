# Multi-label classification, to be fine-tuned

import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTForSequenceClassification.from_pretrained('openai-gpt', problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# ! This is not working
labels = torch.tensor([1, 1], dtype=torch.float) # dtype=float for BCEWithLogitsLoss
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(loss, logits)