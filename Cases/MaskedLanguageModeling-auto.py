# https://huggingface.co/transformers/task_summary.html
from transformers import pipeline

unmasker = pipeline("fill-mask")

from pprint import pprint
pprint(unmasker(f"HuggingFace is creating a {unmasker.tokenizer.mask_token} that the community uses to solve NLP tasks."))

