from einops import rearrange
import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import accuracy_score

import json 
import os.path
from itertools import chain
import statistics

from utils import save_dict_to_json

class TrainerDialogues : 
    def __init__(self, lm_model, model_head, train_data_loader, val_data_loader, test_data_loader, batch_size, n_epoch, criterion, num_epoch_record, config_optim, device, num_channels_post_lm, name_experiment) :
        self.lm_model = lm_model
        self.model_head = model_head
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.criterion = criterion
        self.config_optim = config_optim
        self.num_epoch_record = num_epoch_record
        self.state_dict = {'epoch':{}, 'test':{}}
        self.device = device
        self.num_channels_post_lm = num_channels_post_lm
        self.name_experiment = name_experiment

        self.init_optim()

    def init_optim(self):
        self.optimizer = optim.RAdam(params=chain(self.lm_model.parameters(), self.model_head.parameters()))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.config_optim['max_lr'],
                                                        epochs = self.n_epoch,
                                                        steps_per_epoch= 2 * len(self.train_data_loader),
                                                        pct_start = self.config_optim['pct_start'],
                                                        div_factor = self.config_optim['div_factor'],
                                                        final_div_factor = self.config_optim['final_div_factor'])                 
    def train_epoch(self):
        self.lm_model.train()
        self.model_head.train()
        acc_score = []
        losses = []
        for tokens, labels, mask, max_dialogue_len in self.train_data_loader:
            #print('shape batch after tokenization', tokens['input_ids'].shape)
            tokens = self.lm_model(**(tokens.to(self.device))).last_hidden_state
            #print('tokens post lm shape :', tokens.shape)
            #print('max_dialogue_len:', max_dialogue_len)
            tokens = rearrange(tokens, '(b d) l c -> b d (l c)', d = max_dialogue_len)
            #print('tokens post rearranging :', tokens.shape)
            tokens = self.model_head(tokens, mask)[mask.flatten()]
            loss = self.criterion(tokens, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().tolist())
            acc_score.append(accuracy_score(labels.detach().cpu(), torch.argmax(tokens, dim=-1).detach().cpu()))
        losses = statistics.mean(losses)
        acc_score = statistics.mean(acc_score)
        return losses, acc_score

    def val_epoch(self):
        self.lm_model.eval()
        self.model_head.eval()
        acc_score = []
        losses = []
        with torch.no_grad() : 
            for tokens, labels, mask, max_dialogue_len in self.val_data_loader:
                tokens = self.lm_model(**(tokens.to(self.device))).last_hidden_state
                tokens = rearrange(tokens, '(b d) l c -> b d (l c)', d = max_dialogue_len)
                tokens = self.model_head(tokens, mask)[mask.flatten()]
                loss = self.criterion(tokens, labels)
                losses.append(loss.detach().tolist())
                acc_score.append(accuracy_score(labels.detach().cpu(), torch.argmax(tokens, dim=-1).detach().cpu()))
        losses = statistics.mean(losses)
        acc_score = statistics.mean(acc_score)
        return losses, acc_score
    
    def train(self):
        for epoch in range(self.n_epoch):
            print(f"Epoch {epoch} : ")
            print("Training... ")
            train_loss, train_accuracy = self.train_epoch()
            print(f"Training loss : {train_loss}, Training Accuracy : {train_accuracy}")
            print("Evaluation...")
            val_loss, val_accuracy = self.val_epoch()
            print(f"Evaluation loss : {val_loss}, Evaluation Accuracy : {val_accuracy}")
            self.scheduler.step()
            self.state_dict['epoch'][epoch]= {
                    'train':{
                        'loss' : train_loss, 
                        'accuracy' : train_accuracy
                    }, 
                    'evaluation':{
                        'loss' : val_loss, 
                        'accuracy' : val_accuracy
                    }
                }
            if epoch == 0 :
                self.best_epoch_acc = 0
                self.best_val_acc = val_accuracy
            
            try:
                os.makedirs("results")
            except FileExistsError:
                pass
            save_dict_to_json(self.state_dict, f'results/state_dict_exp_{self.name_experiment}.json')

            if (epoch % self.num_epoch_record == 0 and epoch != 0) or val_accuracy < self.best_val_acc:
                try:
                    os.makedirs(f"checkpoints/model_exp_{self.name_experiment}")
                except FileExistsError:
                    pass
                torch.save({
                'epoch': epoch,
                'lm_model_state_dict': self.lm_model.state_dict(),
                'model_head_state_dict': self.model_head.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'stats' : self.state_dict['epoch'][epoch],
                }, f'checkpoints/model_exp_{self.name_experiment}/epoch_{epoch}.pt')

                if val_accuracy < self.best_val_acc :
                    self.best_epoch_acc = epoch
                    self.best_val_acc = val_accuracy

    def load_model(self, epoch) :
        print('loading')
        checkpoint = torch.load(f'checkpoints/model_exp_{self.name_experiment}/epoch_{epoch}.pt')
        self.lm_model.load_state_dict(checkpoint['lm_model_state_dict'])
        self.model_head.load_state_dict(checkpoint['model_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    def test(self, epoch):
        print("Testing... ")
        '''
        if epoch == None:
            with open(f'results/state_dict_exp_{self.name_experiment}.json') as json_file:
                state_dict = json.load(json_file)
                _best_epoch_acc = 0
                _best_epoch_idx = 0
                for _epoch in state_dict['epoch']:
                    if state_dict['epoch'][_epoch]['evaluation']['accuracy'] > _best_epoch_acc :
                        _best_epoch_idx = _epoch
                        _best_epoch_acc = state_dict['epoch'][_epoch]['evaluation']['accuracy']
        '''
        self.load_model(epoch)
        self.lm_model.eval()
        self.model_head.eval()
        acc_score = []
        losses = []
        with torch.no_grad() : 
            for tokens, labels, mask, max_dialogue_len in self.test_data_loader:
                tokens = self.lm_model(**(tokens.to(self.device))).last_hidden_state
                tokens = rearrange(tokens, '(b d) l c -> b d (l c)', d = max_dialogue_len)
                tokens = self.model_head(tokens, mask)[mask.flatten()]
                loss = self.criterion(tokens, labels)
                losses.append(loss.detach().tolist())
                acc_score.append(accuracy_score(labels.detach().cpu(), torch.argmax(tokens, dim=-1).detach().cpu()))
            losses = statistics.mean(losses)
            acc_score = statistics.mean(acc_score)
            print(print(f"Test loss : {losses}, Test Accuracy : {acc_score}"))
            self.state_dict['test']= {
                'epoch' : epoch, 
                'loss': losses,
                'accuracy' : acc_score
            }                   
            try:
                os.makedirs("test_results")
            except FileExistsError:
                pass
            save_dict_to_json(self.state_dict, f'results/state_dict_exp_{self.name_experiment}.json')

    



            

            


            

        
