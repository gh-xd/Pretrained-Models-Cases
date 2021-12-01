from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = "Hugging Face is based in DUMBO, New York City, and"

inputs = tokenizer(sequence, return_tensors='pt')
input_ids = inputs["input_ids"]

# get logits of last hidden state
next_token_logits = model(**inputs).logits[:,-1,:]

# Filter 
# top_k_top_p_filtering的作用是，把非top-k的Logits转化为负无穷（看作0），top_p -> 过滤到小于它的概率
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)

# multinomial采样函数，根据给定权重对数组进行多次采样，返回元素下标
# torch.multinomial(weights, num_sample, replacement)
# weights 权重，取每个值的概率，1或者2维
# num_samples：采样次数（等于生成个数）
# replacement: default - False，不放回采样

# 我的理解，为什么要采样，因为可以根据softmax最后一层的“概率”分布，来生成不同的输出
next_token = torch.multinomial(probs, num_samples=15)

# cat=concate，拼接，-1自动 = 原始文本 + 生成文本
generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)