import torch
from transformers import GPT2Tokenizer

class Tokenizer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, data):
        #print(type(data), type(data[0]))
        return self.tokenizer(data, padding='max_length', max_length= self.max_length, return_tensors="pt")

def collate_fn(batch):
    data, label = zip(*batch)
    tokenizer = Tokenizer(data)
    return tokenizer.tokenize(data), label