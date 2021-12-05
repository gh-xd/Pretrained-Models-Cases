# LM and Multiple-choice classification

import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
print(tokenizer)
print(len(tokenizer))
model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
# add [CLS] to the vocabulary (it is also trained)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
model.resize_token_embeddings(len(tokenizer))

# Multiple choice
choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"] # batch size = 2
input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0) # 2 choices in batch size > 1
mc_token_ids = torch.tensor([input_ids.size(-1)-1, input_ids.size(-1)-1]).unsqueeze(0) # batch size 1

outputs = model(input_ids, mc_token_ids=mc_token_ids)
# 2 heads on top > 2 output logits
# ! 报错: 'OpenAIGPTDoubleHeadsModelOutput' object has no attribute 'lm_logits'
lm_logits = outputs.lm_logits
mc_logits = outputs.mc_logits

# 能做啥？


