from transformers import AutoTokenizer
from pprint import pprint
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 01 - Single sentence encoding and decoding
encoded_input = tokenizer("Hello, I'm a single sentence!")
pprint(encoded_input)

decoded_input = tokenizer.decode(encoded_input["input_ids"])
pprint(decoded_input)

# 02 - batch sentence encoding and decoding
batch_sentences = ["Hello I'm a single sentence", "And another sentence", "And the very very last one"]
encoded_input = tokenizer(batch_sentences)
pprint(encoded_input)

# 03 - but the encoding in 02 is not fixed length --> if input is a batch --> padding and truncate
batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt") # for models having max length
pprint(batch)


# 04 - BERT model add automatically [CLS]...[SEP]...[SEP]
encoded_input = tokenizer("How old are you?", "I'm 6 years old")
# 'token_type_ids' means 1st or 2nd sentence
pprint(encoded_input)

# take a look at decoding
decoded_input = tokenizer.decode(encoded_input["input_ids"])
pprint(decoded_input)

# 05 - list pair of sentences --> just input, it will work
batch_sentences = ["Hello I'm a single sentence", "And another sentence", "And the very very last one"]
batch_of_second_sentences = ["I'm a sentence that goes with the first sentence","And I should be encoded with the second sentence", "And I go with the very last one"]

encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
pprint(encoded_inputs)

# decoding the sentences
for ids in encoded_inputs["input_ids"]:
    pprint(tokenizer.decode(ids))

# 06 - Pre-tokenized inputs (if tokens exist for e.g. 1) NER or 2) POS tagging - part-of-speech tagging)
encoded_input = tokenizer(["Hello", "I'm", "a", "single", "sentence"], is_split_into_words=True) 
pprint(encoded_input)

# same for batch
batch_sentences = [["Hello", "I'm", "a", "single", "sentence"],
                   ["And", "another", "sentence"],
                   ["And", "the", "very", "very", "last", "one"]]
# encoded_inputs = tokenizer(batch_sentences, is_split_into_words=True)

batch_of_second_sentences = [["I'm", "a", "sentence", "that", "goes", "with", "the", "first", "sentence"],
                             ["And", "I", "should", "be", "encoded", "with", "the", "second", "sentence"],
                             ["And", "I", "go", "with", "the", "very", "last", "one"]]
encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences, is_split_into_words=True)
pprint(encoded_inputs)