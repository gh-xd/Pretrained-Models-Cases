# https://huggingface.co/transformers/task_summary.html
from transformers import pipeline

# download a model automatically, without specification
classifier = pipeline("sentiment-analysis")

test1 = classifier("I hate you")[0]
print(f"label: {test1['label']}, with score: {round(test1['score'], 4)}")

test2 = classifier("I love you")[0]
print(f"label: {test2['label']}, with score: {round(test2['score'], 4)}")
