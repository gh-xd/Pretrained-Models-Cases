from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Classification():
    def __init__(self, model, cache_dir='./model_cache'):
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir)
    def nsp(self, sent_1, sent_2):
        sent_pair = self.tokenizer(sent_1, sent_2, return_tensors='pt')
        sent_logits = self.model(**sent_pair).logits
        sent_pair_res = torch.softmax(sent_logits, dim=1).tolist()[0]
        res_dict = {'not paraphrase': round(sent_pair_res[0]), 'is paraphrase':round(sent_pair_res[1])}
        return res_dict
