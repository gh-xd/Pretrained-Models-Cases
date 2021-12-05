from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# one for sentence pridiction, one for nsp
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits

print(prediction_logits,"\n")
print(seq_relationship_logits)