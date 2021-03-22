#imports
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, RobertaTokenizer
import json
import re
import pandas as pd
import os


# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

class StandardScalar:
    def __init__(self):
        """
        Can be updated by calling the fit functionality with train data.
        """
        self.means = []
        self.stds = []
    
    def fit(self, x=[], means=[], stds=[]):
        """
        x : list of list num_data X num_feats
        #compute means and stds from x OR use supplied values.
        """
        if(len(means) and len(stds)):
            self.means = means
            self.stds = stds
        else:
            assert len(x) > 0
            x = np.array(x)
            #print(x.shape)
            self.means = np.array([np.mean(x[:,i]) for i in range(x.shape[1])]) #i: every feature
            self.stds = np.array([np.std(x[:,i]) for i in range(x.shape[1])]) #i: every feature
            #print("Means, stds: ", self.means, self.stds)
    
    def transform(self, x):
        """
        standaridize x with mean 0 and std 1
        """
        assert len(self.means)*len(self.stds) > 0

        return np.array(x).astype(np.float32) # do not scale.
        #return ((np.array(x) - self.means)/self.stds).astype(np.float32)

"""
Class to handle all stages before model training: data pre-processing + feature engineering.
"""
class DataHandler:

    def __init__(self, args, logger):
        """
        Initiating Data Handler object.
        """
        self.args = args
        self.logger = logger
        self.scalar = StandardScalar()
        self.pos_matrix = positional_encoding(100, 32)[0].tolist() # 100 tokens , 32 dimensions.

        #update args from checkpoint, if asked to.
        if(self.args.load_from_ckpt):
            
            store_logdir = self.args.logdir
            ckptfile = os.path.join(self.args.logdir, self.args.ckpt_file)
            print("Loading arguments and scalar values from checkpoint file: ", ckptfile, file=self.logger)
            ckpt = torch.load(ckptfile)
            self.args.update_hyps(ckpt["args"])
            
            self.args.load_from_ckpt = True
            self.args.logdir = store_logdir
            #update scalar.
            self.scalar.fit(means = ckpt["means"], stds = ckpt["stds"]) 
            print("Loaded args: ", vars(self.args), file=self.logger)
        
        if(self.args.bert):

            if(self.args.pretraining):
                assert self.args.pretrained_dir
                self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_dir)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(PATH_TO_MODIFIED_INPUT_DIR)

    def get_label_rep(self, label_string):

        labels = self.args.labels
        
        assert self.args.num_labels == len(labels)
        found = set(label_string.split(','))
        label_rep = [0 for _ in range(len(labels))]

        for i, label in enumerate(labels):
            if(label in found):
                label_rep[i]= 1
        
        return label_rep

    def get_initial_io(self, all_data, train=False):
        """
        x: (context, sentence, feats)
        feats can be (turn embedding, bow of context annotations)
        y: (basically list of list of trues, given a sequence of labels)
        """
        initial_io = {

            'context': [],
            'utterance': [],
            'feat': [],
            'y': []
        }

        for dialogue in all_data:
            for i, item in enumerate(dialogue):

                utterance = item[0]

                context_items = dialogue[max(i-self.args.context_size, 0): i]
                
                context = [ii[0] for ii in context_items]
                this_context = context[:]

                this_utterance = utterance

                #features
                features = self.pos_matrix[i][:] #[i] #turn number
                assert len(features) == 32, len(features)
                context_bow = [self.get_label_rep(ii[1]) for ii in context_items]
                merged_bow = [0 for _ in range(self.args.num_labels)]
                for bow in context_bow:
                    for ix in range(self.args.num_labels):
                        merged_bow[ix] += bow[ix]
                #features += merged_bow #annotations of the context
                this_features = features[:]

                #y
                this_y = self.get_label_rep(item[1])[:]

                #add to the array
                initial_io['context'].append(this_context[:])
                initial_io['utterance'].append(this_utterance)
                initial_io['feat'].append(this_features[:])
                initial_io['y'].append(this_y[:])

                if(self.args.oversample and train):
                    #do over sampling for minority class:  this_y[3]==1 or this_y[5]==1
                    if(sum(this_y) >= 1):
                        #atleast one positive sample
                        #add to the array
                        initial_io['context'].append(this_context[:])
                        initial_io['utterance'].append(this_utterance)
                        initial_io['feat'].append(this_features[:])
                        initial_io['y'].append(this_y[:])
        
        return initial_io

    def get_tokens(self, cxt, utt):
        """
            sent = "[CLS] " + " ".join(cxt) + "[SEP] " + utt + "[SEP]"
            tokens = self.tokenizer.tokenize(sent)
            
            if(len(tokens) >= self.args.max_tokens):
                tokens = tokens[:self.args.max_tokens]
            else:
                tokens = tokens + ['[PAD]' for _ in range(self.args.max_tokens-len(tokens))]
        """
        def get_num_toks(all_toks):
            summ = 0
            for item in all_toks:
                summ += len(item)                
            return summ

        all_toks = [self.tokenizer.tokenize(sent) for sent in cxt] + [self.tokenizer.tokenize(utt)]

        count = get_num_toks(all_toks)
        rem = count - (self.args.max_tokens - 20)
            
        for i in range(len(all_toks)):
            if(rem <= 0):
                break
            length = len(all_toks[i])
            if(length >= rem):
                all_toks[i] = all_toks[i][rem:]
                rem = 0
            else:
                all_toks[i] = []
                rem -= length

        if(len(all_toks) < (self.args.context_size + 1)):
            missing = (self.args.context_size + 1) - len(all_toks)
            all_toks = [[] for _ in range(missing)] + all_toks
        
        final_tokens = ['[CLS]']
        for item in all_toks[:-1]:
            final_tokens += item
            final_tokens.append('<end_of_text>') #ADD TOKEN TO VOCABULARY.
        
        final_tokens.append('[SEP]')

        final_tokens += all_toks[-1]
        final_tokens.append('[SEP]')

        if(len(final_tokens) >= self.args.max_tokens):
            final_tokens = final_tokens[:self.args.max_tokens]
        else:
            final_tokens = final_tokens + ['[PAD]' for _ in range(self.args.max_tokens-len(final_tokens))]

        assert len(final_tokens) == self.args.max_tokens
        return final_tokens

    def make_ready_for_input(self, all_data, fit_scalar):
        """
        x: x_input_ids, x_input_type_ids, x_input_mask, x_input_feats
        y: y
        """
        input_ready = {
            "x_input_ids": [],
            "x_input_type_ids": [],
            "x_input_mask": [],
            "x_input_feats": [],
            "y": []
        }
        #features
        if(fit_scalar):
            self.scalar.fit(x=all_data["feat"])
        input_ready['x_input_feats'] = self.scalar.transform(all_data["feat"])
        
        ix = 0
        #other inputs
        for cxt, utt in zip(all_data["context"], all_data["utterance"]):
            tokens = self.get_tokens(cxt, utt)
            if(ix == 1):
                #random printing
                print(tokens, file=self.logger)
            ix += 1

            ids = self.tokenizer.convert_tokens_to_ids(tokens)

            x_input_type_ids = []
            x_input_mask = []

            type_cur = 0
            for tok in tokens:
                
                x_input_type_ids.append(type_cur)
                if(tok == '[SEP]'):
                    type_cur = 1
                
                if(tok == '[PAD]'):
                    x_input_mask.append(0)
                else:
                    x_input_mask.append(1)
            input_ready["x_input_ids"].append(ids)
            input_ready["x_input_type_ids"].append(x_input_type_ids)
            input_ready["x_input_mask"].append(x_input_mask)
        
        #y
        input_ready["y"] = all_data["y"]

        print(input_ready["x_input_ids"][1],
              input_ready["x_input_type_ids"][1],
              input_ready["x_input_mask"][1],
              input_ready["x_input_feats"][1],
              input_ready["y"][1],
             file=self.logger)
        
        return input_ready

    def make_tensors(self, all_data):
        
        all_data["x_input_ids"] = torch.tensor(all_data["x_input_ids"], dtype=torch.long)
        all_data["x_input_type_ids"] = torch.tensor(all_data["x_input_type_ids"], dtype=torch.long)
        all_data["x_input_mask"] = torch.tensor(all_data["x_input_mask"], dtype=torch.long)
        all_data["x_input_feats"] = torch.tensor(all_data["x_input_feats"])
        all_data["y"] = torch.tensor(all_data["y"], dtype=torch.float32)

        return all_data

    def get_dataloader(self, all_data, fit_scalar=False, use_random_sampler=True, train=False):
        """
        Data format: 
        Basic format of input all_data is dialogue with (, separated) annotation labels: go from there.
        """
        
        """
        x: (context, utterance, feats)
        feats can be (turn embedding, bow of context annotations)
        y: (basically list of list of trues, given a sequence of labels)
        """
        all_data = self.get_initial_io(all_data, train=train)

        """
        x: x_input_ids, x_input_type_ids, x_input_mask, x_input_feats
        y: y
        """
        all_data = self.make_ready_for_input(all_data, fit_scalar)

        """
        convert stuff to tensors
        """
        all_data = self.make_tensors(all_data)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory
        data = TensorDataset(all_data["x_input_ids"], all_data["x_input_type_ids"], all_data["x_input_mask"], all_data["x_input_feats"], all_data["y"])

        if(use_random_sampler):
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.args.batch_size)
        
        return dataloader