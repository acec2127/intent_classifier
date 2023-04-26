import torch
import numpy as np
import random
from itertools import chain
from tokenizer import Tokenizer

class SiliconeDataloader :
    def __init__(self, dataset, batch_size, max_utterance_len, device):
        self.batch_size = batch_size

        self.utterances = dataset.utterances
        self.labels = dataset.labels
        self.dialogues = dataset.dialogues
        self.tokenizer = Tokenizer(max_utterance_len)
        self.device = device

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self) :
        random.shuffle(self.dialogues)
        count = 0
        batch_indices = []
        batch_len = []
        for dialogue in self.dialogues:
            if count + dialogue[1] > self.batch_size :
                batch_max_dialogue_len = max(batch_len)
                batch_utterances = []
                batch_labels = []
                masks = np.zeros((len(batch_len), batch_max_dialogue_len), dtype=np.bool_)
                for i, (dialogue_idx, dialogue_len) in enumerate(zip(batch_indices, batch_len)):
                    batch_utterances.append(self.utterances[dialogue_idx:dialogue_idx+dialogue_len] + [""]*(batch_max_dialogue_len - dialogue_len))
                    batch_labels.append(self.labels[dialogue_idx:dialogue_idx+dialogue_len])
                    masks[i,:dialogue_len] = True

                yield self.tokenizer.tokenize(list(chain.from_iterable(batch_utterances))).to(self.device),\
                    torch.tensor(list(chain.from_iterable(batch_labels)), dtype=torch.uint8).to(self.device),\
                    torch.from_numpy(masks).to(self.device), batch_max_dialogue_len
                count = 0
                batch_indices = []
                batch_len = []

            batch_indices.append(dialogue[0])
            batch_len.append(dialogue[1])
            count += dialogue[1]

        if batch_len :
            batch_max_dialogue_len = max(batch_len)
            batch_utterances = []
            batch_labels = []
            masks = np.zeros((len(batch_len), batch_max_dialogue_len), dtype=np.bool_)
            for i, (dialogue_idx, dialogue_len) in enumerate(zip(batch_indices, batch_len)):
                batch_utterances.append(self.utterances[dialogue_idx:dialogue_idx+dialogue_len] + [""]*(batch_max_dialogue_len - dialogue_len))
                batch_labels.append(self.labels[dialogue_idx:dialogue_idx+dialogue_len])
                masks[i,:dialogue_len] = True

            yield self.tokenizer.tokenize(list(chain.from_iterable(batch_utterances))).to(self.device),\
                torch.tensor(list(chain.from_iterable(batch_labels)), dtype=torch.uint8).to(self.device),\
                torch.from_numpy(masks).to(self.device), batch_max_dialogue_len