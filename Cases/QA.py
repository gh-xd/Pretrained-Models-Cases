from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QA():
    def __init__(self, model, cache_dir='./model_cache/'):
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model, cache_dir=cache_dir)
    def answer(self, questions, facts):
        if isinstance(questions, list):
            pass
        else:
            print(">> Change input type to list")

        answer_list = []
        for question in questions:
            inputs = self.tokenizer(question, facts, add_special_tokens=True, return_tensors='pt')
            input_ids = inputs["input_ids"].tolist()[0]

            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            answer_list.append(answer)
            
        return answer_list

if __name__ == '__main__':
    questions = [
    'How many pretrained models are available in HuggingFace Transformers?',
    'What doese HuggingFace Transformers provide?',
    'HuggingFace Transformers provides interoperability between which frameworks?'
]

    text = r"""
            HuggingFace Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
            architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
            Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
            TensorFlow 2.0 and PyTorch."""

    qa = QA(model="bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir='./QA/')
    answers = qa.answer(questions=questions, facts=text)
    print(answers)