# Single-label classification, to be fine-tuned
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTForSequenceClassification.from_pretrained('openai-gpt')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0) # batch size > 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(loss, logits)