from transformers import OpenAIGPTTokenizer, OpenAIGPTModel

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTModel.from_pretrained('openai-gpt')

inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
outputs = model(**inputs)


# Raw transformer outputs
last_hidden_states = outputs.last_hidden_state

# shape = (batch size=1, sequence_size=7, vec_size=768)
print(last_hidden_states.shape)

