from transformers import GPT2Tokenizer, GPT2ForTokenClassification
import torch

tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialogRPT-updown')
model = GPT2ForTokenClassification.from_pretrained('microsoft/DialogRPT-updown')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# token x n
labels = torch.tensor([1]*inputs["input_ids"].size(1)).unsqueeze(0) # Batch size 1

outputs = model(**inputs, labels=labels)
loss = outputs
logits = outputs.logits

print(loss, logits)