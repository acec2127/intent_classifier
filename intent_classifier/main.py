import json
import click

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM

from model_head import ModelHead
from trainer import TrainerDialogues

from dataset import SiliconeDataset
from dataloader import SiliconeDataloader

@click.command()
@click.option('--exp', required=True, type=int)
@click.option('--ep', required=True, type=int)

def main(exp, ep):
    with open(f'experiments/experiment_{exp}.json') as json_file:
        config = json.load(json_file)

    dataset_name = config['details']['dataset']
    batch_size = config['dataloader']['batch_size']
    #only dialogues ones
    assert dataset_name in [
        'dyda_da', 'dyda_e',
        'meld_e', 'meld_s', 'mrda', 'oasis', 'sem', 'swda'
    ]
    dataset = load_dataset("silicone", dataset_name)

    train_data = SiliconeDataset(dataset, 'train', batch_size)
    val_data = SiliconeDataset(dataset, 'validation', batch_size)
    test_data = SiliconeDataset(dataset, 'test', batch_size)

    max_utterance_len = train_data.max_utterance_len
    num_channels = 768
    num_channels_post_lm = max_utterance_len*num_channels
    n_class = train_data.n_class
    print('Max utterance len :', max_utterance_len)

    print('Device is cuda ? : ', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    lm_model = lm_model.transformer.to(device)
    lm_model.config.pad_token_id = lm_model.config.eos_token_id

    model_head = ModelHead(
        config['model_head']['num_layers'],
        config['model_head']['num_heads'],
        num_channels_post_lm,
        config['model_head']['num_channels'],
        n_class,
        widening_factor = config['model_head']['widening_factor'],
        dropout = config['model_head']['dropout']
    ).to(device)

    '''
    assert(max(train_data.dialogues[1]) <= batch_size),\
        f"batch_size {batch_size} too small, max dialogue len is {max(train_data.dialogues[:, 1])}"
    assert(max(val_data.dialogues[:, 1]) <= batch_size),\
        f"batch_size {batch_size} too small, max dialogue len is {max(val_data.dialogues[:, 1])}"
    assert(max(test_data.dialogues[:, 1]) <= batch_size),\
        f"batch_size {batch_size} too small, max dialogue len is {max(test_data.dialogues[:, 1])}"
    '''

    train_dataloader = SiliconeDataloader(train_data, batch_size,\
        max_utterance_len, device)
    
    val_dataloader = SiliconeDataloader(val_data, batch_size,\
        max_utterance_len, device)
    
    test_dataloader = SiliconeDataloader(test_data, batch_size,\
        max_utterance_len, device)
    
    trainer = TrainerDialogues(
        lm_model, model_head, train_dataloader, val_dataloader, test_dataloader, batch_size,
        config['dataloader']['n_epoch'], nn.CrossEntropyLoss(), config['dataloader']['num_epoch_record'],
        config['optimizer'], device, num_channels_post_lm, config['details']['experiment']
    )
    #trainer.train()
    trainer.test(ep)

    return 

if __name__ == '__main__' : 
    main()







