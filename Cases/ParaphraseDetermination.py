# https://huggingface.co/transformers/task_summary.html
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model (download: ~434M)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", cache_dir='./ParagraphDetermination/')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", cache_dir='./ParagraphDetermination/')

classes = ["not paraphrase", "is paraphrase"]

# text sample
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# Use tokenizer preprocess sentence pair
sentence_pair_1 = tokenizer(sequence_0, sequence_2, return_tensors='pt')
sentence_pair_2 = tokenizer(sequence_0, sequence_1, return_tensors='pt')

# send pre-processed sentence pair into model (bert-base-cased) for prediction logits (vectors)
# logits shape here (1,2)
sp1_logits = model(**sentence_pair_1).logits
sp2_logits = model(**sentence_pair_2).logits

# softmax the prediction logits (vectors) to get the predicted class
sp1_results = torch.softmax(sp1_logits, dim=1).tolist()[0]
sp2_results = torch.softmax(sp2_logits, dim=1).tolist()[0]

# Take a look at predicted classes for sentence pair 1
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(sp1_results[i] * 100))}%")

# Take a look at predicted classes for sentence pair 2
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(sp2_results[i] * 100))}%")

# Results show that 1) sentence 0 and 2 are closer; 2) sentence 0 and sentence 2 are not related