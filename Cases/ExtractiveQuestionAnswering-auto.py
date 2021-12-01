# https://huggingface.co/transformers/task_summary.html
from transformers import pipeline

question_answer = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script."""

result_1 = question_answer(question="What is extractive question answering", context=context)
print(f"Answer: {result_1['answer']}, score: {round(result_1['score'], 4)}, start: {result_1['start']}, end: {result_1['end']}")

result_2 = question_answer(question="What is a good example of a question answering dataset", context=context)
print(f"Answer: {result_2['answer']}, score: {round(result_2['score'], 4)}, start: {result_2['start']}, end: {result_2['end']}")
