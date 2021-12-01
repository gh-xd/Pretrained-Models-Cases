# https://huggingface.co/transformers/task_summary.html#text-generation
# default gpt2
from transformers import pipeline

text_generator = pipeline("text-generation")

print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))