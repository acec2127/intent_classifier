import numpy as np
#from torch.utils.data import Dataset

class SiliconeDataset:
    def __init__(self, data, mode, batch_size) :
        self.batch_size = batch_size
        if mode == 'train':
            self.n_class = max(data["train"]['Label'] + data["validation"]['Label'] + data["test"]['Label']) + 1
            self.max_utterance_len = len(max( data['train']['Utterance'] + data["validation"]['Utterance'] + data["test"]['Utterance'], key=len))

        self.utterances = data[mode]['Utterance']
        self.labels = data[mode]['Label']
        self.dialogue_id = list(map(int, data[mode]['Dialogue_ID']))
        self.get_dialogue_idx()

    def __len__(self):
        return len(self.labels)
    
    def get_dialogue_idx(self):
        dialogue_idx = [0]
        dialogue_len = []
        start_index = self.dialogue_id[0]
        count = 0
        for idx, index in enumerate(self.dialogue_id) :
            if index != start_index or count == self.batch_size:
                #assert(len(self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count]) == count),\
                #f"count: {count},string {self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count]}, string_len:{len(self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count])}"
                dialogue_idx.append(idx)
                dialogue_len.append(count)
                start_index = index
                count = 0
            count += 1
        #assert(len(self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count]) == count),\
        #        f"count: {count},string {self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count]}, string_len:{len(self.utterances[dialogue_idx[-1]:dialogue_idx[-1]+count])}"
        
        #print('final count', count)
        dialogue_len.append(count)

        assert(len(dialogue_len) == len(dialogue_idx) ),\
        f"{(len(dialogue_len), len(dialogue_idx))}"

        self.dialogues=list(zip(dialogue_idx, dialogue_len))
        #print('dialogues', self.dialogues)