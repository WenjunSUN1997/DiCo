from datasets import load_dataset
# raw_datasets = load_dataset("code_search_net", "python")
# print(raw_datasets["train"])
# print(raw_datasets["train"][123456]["whole_func_string"])
# def get_training_corpus():
#     return (
#         raw_datasets["train"][i : i + 1000]["whole_func_string"]
#         for i in range(0, len(raw_datasets["train"]), 1000)
#     )
#
#
# training_corpus = get_training_corpus()
# a = [x for x in training_corpus]
# from transformers import AutoTokenizer
#
# old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# example = '''def add_numbers(a, b):
#     """Add the two numbers `a` and `b`."""
#     return a + b'''
#
# tokens = old_tokenizer.tokenize(example)
# print(tokens)
# tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
# tokens = tokenizer.tokenize(example)
# print(tokens)

from datasets import load_dataset, load_metric
raw_datasets = load_dataset("wmt16", "ro-en")['train']
raw_datasets.to_csv('test.csv')