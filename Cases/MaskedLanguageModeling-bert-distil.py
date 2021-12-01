from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", cache_dir="./cache/")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased", cache_dir="./cache/")

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would be help {tokenizer.mask_token} our carbon footprint."

inputs = tokenizer(sequence, return_tensors='pt')
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

token_logits = model(**inputs).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

# Result: 5 generation for masked position