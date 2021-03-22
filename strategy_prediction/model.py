#imports
from transformers import AdamW, BertForSequenceClassification, BertModel, RobertaModel, BertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, max_error, accuracy_score, f1_score, classification_report
import torch.optim as optim
import copy

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

"""
Core PyTorch network
"""
class Net(nn.Module):
    def __init__(self, args):
        """
        Defining the layers in the architecture.
        """
        super(Net, self).__init__()
        
        self.args = args
        self.total_num_hidden = 0

        if(self.args.text_encoder):
            #Final embedding size will be self.args.text_output_dims
            self.total_num_hidden += self.args.text_output_dims

            if(self.args.bert):
                if(self.args.pretraining):
                    assert self.args.pretrained_dir
                    self.bert_layer = BertModel.from_pretrained(self.args.pretrained_dir) 
                    print("loaded from pretrained model")
                else:   
                    self.bert_layer = BertModel.from_pretrained(PATH_TO_MODIFIED_INPUT_DIR)
 
            #TODO BERT tokenizer, scratch, Bi-LSTM
    
            #task-specific transformer layers
            self.task_trans = nn.ModuleList([self.get_task_trans_model() for _ in range(self.args.num_labels)])

            #Feed-forward, activation, dropout on the output of task_trans layers.
            self.after_trans_fcs = nn.ModuleList([nn.Linear(self.args.text_encoder_dims, self.args.text_output_dims) for _ in range(self.args.num_labels)])
            self.after_trans_dropouts = nn.ModuleList([nn.Dropout(self.args.dropout) for _ in range(self.args.num_labels)])
            self.after_trans_activations = nn.ModuleList([self.get_activation_layer() for _ in range(self.args.num_labels)]) 

        if(self.args.feature_encoder):
            #self.total_num_hidden += self.args.feature_encoder_dims
            self.feature_fc = nn.Linear(self.args.feature_input_dims, self.args.feature_encoder_dims)
            self.feature_activation = self.get_activation_layer()
            self.feature_dropout = nn.Dropout(self.args.dropout)
        
        #task-specific final layers
        self.task_fcs1 = nn.ModuleList([nn.Linear(self.total_num_hidden, self.total_num_hidden//2) for _ in range(self.args.num_labels)])
        self.task_fcs2 = nn.ModuleList([nn.Linear(self.total_num_hidden//2, 1) for _ in range(self.args.num_labels)])
        self.task_dropouts = nn.ModuleList([nn.Dropout(self.args.dropout) for _ in range(self.args.num_labels)])
        self.task_activations = nn.ModuleList([self.get_activation_layer() for _ in range(self.args.num_labels)])
    
    def get_activation_layer(self):
        if(self.args.activation == 'relu'):
            return nn.ReLU()
        elif(self.args.activation == 'linear'):
            return nn.Identity()
        
        #should not reach here
        raise AssertionError

    def get_task_trans_model(self):
        
        config = BertConfig()
        config.num_hidden_layers=1
        config.num_attention_heads=1
        config.hidden_size=self.args.text_encoder_dims

        return BertModel(config)

    def apply_task_trans(self, trans_model, X_bert, x_input_mask, length):
        X_this = trans_model(inputs_embeds=X_bert, attention_mask=x_input_mask)
        X_this = X_this[1]
        assert X_this.shape == (length, self.args.text_encoder_dims)

        return X_this

    def after_trans_layers(self, x, fc, activation, dropout, length):

        X_this = fc(x)
        X_this = activation(X_this)
        X_this = dropout(X_this)

        assert X_this.shape == (length, self.args.text_output_dims)

        return X_this

    def apply_final_layers(self, fcs1, fcs2, dropout, activation, x_input, length):

        assert x_input.shape == (length, self.total_num_hidden)     
        X_this = fcs1(x_input)
        X_this = activation(X_this)
        X_this = dropout(X_this)

        X_this = fcs2(X_this)
        assert X_this.shape == (length, 1)

        return X_this
    
    def forward(self, x_input_ids=None, x_input_type_ids=None, x_input_mask=None, x_input_feats=None):
        """
        Define the forward pass here.
        
        x_input_ids = (batch_size, max_tokens)
        x_input_type_ids = (batch_size, max_tokens)
        x_input_mask = (batch_size, max_tokens)
        """

        #true batch size for this batch
        length = x_input_ids.shape[0]
        
        X = []

        if(self.args.text_encoder):
            assert x_input_ids.shape == (length, self.args.max_tokens) and x_input_ids.shape == x_input_mask.shape and x_input_ids.shape == x_input_type_ids.shape

            if(self.args.bert):

                #apply bert_layer
                X_bert = self.bert_layer(x_input_ids, token_type_ids=x_input_type_ids, attention_mask=x_input_mask)

                if(self.args.have_attn):
                    X_bert = X_bert[0]
                    assert X_bert.shape == (length, self.args.max_tokens, self.args.text_encoder_dims)
                else:
                    #no attn ...choose the CLS.
                    X_bert = X_bert[1]
                    assert X_bert.shape == (length, self.args.text_encoder_dims)

            if(self.args.have_attn):
                #apply task specific trans layers: only makes sense when you are encoding the text.
                Xis = [self.apply_task_trans(item, X_bert, x_input_mask, length) for item in self.task_trans]
            else:
                Xis = [torch.clone(X_bert) for _ in range(self.args.num_labels)] # no attn layer, that means, just the BERT CLS used multiple times..
            
            #apply further layers.
            Xis = [self.after_trans_layers(x, fc, activation, dropout, length) for x, fc, activation, dropout in zip(Xis, self.after_trans_fcs, self.after_trans_activations, self.after_trans_dropouts)]
            X = Xis

        if(self.args.feature_encoder):
            assert x_input_feats.shape == (length, self.args.feature_input_dims)
            X_feature = self.feature_fc(x_input_feats)
            X_feature = self.feature_activation(X_feature)
            X_feature = self.feature_dropout(X_feature)
            assert X_feature.shape == (length, self.args.feature_encoder_dims)

            if(not len(X)):
                X = [torch.clone(X_feature) for _ in range(self.args.num_labels)]
            else:
                X = [(item + torch.clone(X_feature)) for item in X] #just add them, they have already been multipled by weight matrices. 
                #X = [torch.cat((item, X_feature), dim=1) for item in X]

        #Finally, run fully connected layers for every task separetely.

        X = [self.apply_final_layers(self.task_fcs1[i], self.task_fcs2[i], self.task_dropouts[i], self.task_activations[i], X[i], length) for i in range(self.args.num_labels)]
        
        #combine the logits
        X = torch.cat(X, dim=1)
        assert X.shape == (length, self.args.num_labels)
        return X
    
"""
Class to handle model training/evaluations and print out corresponding stats.
"""
class ModelHandler:

    def __init__(self, data, args, logger):
        """
        data: formatted data, after feature engineering. keys: *_x/y/feats, *:train, dev, test.
        args: command line arguments
        If the mode is eval or we are training from a checkpoint, assume that the associated arguments are already loaded up correctly.
        """
        self.data = data
        self.args = args
        self.logger = logger

        #define the model graph and load any pre-trained embeddings.
        self.build_model()

        #load model from ckpt if asked for: This can either be used to train from a specific point or for evaluating a checkpoint, depending on what mode the code is running in.
        if(self.args.load_from_ckpt):
            ckptfile = os.path.join(self.args.logdir, self.args.ckpt_file)
            print("Loading model from checkpoint file: ", ckptfile, file=self.logger)
            ckpt = torch.load(ckptfile)
            self.model.load_state_dict(ckpt["ckpt"])

    def build_model(self):
        """
        Build the model graph and set it up for upcoming training or evaluation.
        """
        
        self.model = Net(self.args)
        #print(self.model, file=self.logger)

        if(self.args.freeze):
            print("FREEZING", file=self.logger)
            print(self.model.bert_layer.parameters(), file=self.logger)
            for param in self.model.bert_layer.parameters():
                param.requires_grad = False
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device, file=self.logger)
        self.model.to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.training_loss = nn.BCEWithLogitsLoss() #multi-label classification..targets are real numbers between 0 and 1, can give a rescaling weight to each class.

    def train_model(self):
        """
        train the model.
        return mae, ckpt and summary for best checkpoint, as per the validation performance and summary for plotting.
        """
        
        #train summary for plotting, analysis.
        summary = {
            'iter_level_train_loss': [], #stored for the batch after every iteration
            'iter_level_dev_loss': [], #stored for one randomly sampled batch after every iteration
            'train_loss': [], #stored for complete training set after every evaluation on train set
            'dev_loss': [], #stored for complete dev set after every evaluation on validation set
            'train_f1': [], #stored for complete train set after every evaluation on train set
            'dev_f1': [] #stored for complete dev set after every evaluation on validation set
        }

        # Set our model to training mode (as opposed to evaluation mode)
        self.model.train()
        
        iter_no = 0
        best_f1 = -1
        best_ckpt = None

        for _ in range(self.args.num_epochs):
            
            # Train the model for one epoch
            for batch in self.data.train_dataloader:
                
                if(iter_no%self.args.print_iter_no_after == 0):
                    print("iter_no: ", iter_no, file=self.logger)
                    self.logger.flush()
                    
                if((iter_no%self.args.ckpt_after == 0) and (iter_no != 0)):
                    #evaluate on dev and update best stuff if found
                    train_loss, train_f1 = self.evaluate_model("train", sample_size=-1, during_training=True)
                    dev_loss, dev_f1 = self.evaluate_model("dev", sample_size=-1, during_training=True)
                    
                    #save summary
                    summary["train_loss"].append(train_loss)
                    summary["dev_loss"].append(dev_loss)
                    summary["train_f1"].append(train_f1)
                    summary["dev_f1"].append(dev_f1)
                    
                    print("iter_no, train_loss, dev_loss, train_f1, dev_f1: ", iter_no, train_loss, dev_loss, train_f1, dev_f1, file=self.logger) 
                    
                    #check if better
                    if(dev_f1 >= best_f1):
                        print("Better model found. dev_f1, best_f1: ", dev_f1, best_f1, file=self.logger)
                        best_f1 = dev_f1
                        best_ckpt = copy.deepcopy(self.get_ckpt())
                        
                    #flush the output
                    self.logger.flush()
                
                #max iter numbers
                if(iter_no >= self.args.num_iters):
                    print("Maximum iters reached, stopping.", file=self.logger)
                    break
                    
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, y = batch
                
                # Clear out the gradients (by default they accumulate)
                self.optimizer.zero_grad()
                
                # Forward pass
                outs = self.model(x_input_ids, x_input_type_ids, x_input_mask, x_input_feats)
                assert outs.shape == y.shape
                loss = self.training_loss(outs, y)
                
                # Backward pass
                loss.backward()
                
                #grad clipping outside the optimizer since transformers update
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                # Update parameters and take a step using the computed gradient
                self.optimizer.step()
                
                #save train loss on this batch
                summary['iter_level_train_loss'].append(loss.item())

                #compute validation loss for a single random batch
                dev_loss_one_iter, _ = self.evaluate_model("dev", sample_size=1, during_training=True)
                summary['iter_level_dev_loss'].append(dev_loss_one_iter)
                
                #update iter number
                iter_no += 1
                
            #max iter numbers
            if(iter_no >= self.args.num_iters):
                print("Maximum iters reached, stopping.", file=self.logger)
                break
                
        #return stored stuff
        return best_f1, best_ckpt, summary

    def evaluate_model(self, data_type, sample_size, during_training=False):
        """
        Evaluate the model, assuming the correct checkpoint is already loaded.
        Return stuff depending on the value of during_training.

        Basically, we need preds and trues for every class separately.
        Then compute
        mean loss, 
        accuracy, classification report for each class....and f1 score for class=1, mean accuracy over all labels, mean f1 (for positive class) over all labels.

        if during_training:
            return loss, mean f1 for positive class
        else:
            return accuracy, classification reports, positive class f1 for each class, avg accuracy, avg f1.
        """
        
        # Set our model to eval mode
        self.model.eval()
        
        loss_sum = 0.0 #total sum of loss.
        total = 0 #number of data points
        preds, trues = [], [] #each item is a list of num_labels.
        
        if(data_type == "train"):
            mydataloader = self.data.train_dataloader
        elif(data_type == "dev"):
            mydataloader = self.data.dev_dataloader
        else:
            mydataloader = self.data.test_dataloader
        
        for step, batch in enumerate(mydataloader):
            
            if((sample_size != -1) and (step >= sample_size)):
                break

            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, y = batch
            length = x_input_ids.shape[0]
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass
                outs = self.model(x_input_ids, x_input_type_ids, x_input_mask, x_input_feats)
            assert outs.shape == y.shape
            loss = self.training_loss(outs, y)
            loss_sum += loss.item()*length#we need the sum not mean here, we will take global mean later.
            total += length

            b_logits = outs.to("cpu").numpy().tolist() #should be a list of lists, each with num_labels classes.
            b_trues = y.to("cpu").numpy().tolist()
            
            assert (len(b_logits) == len(b_trues)) and (len(b_logits[0]) == len(b_trues[0]) == self.args.num_labels)

            if(self.args.majority_prediction):
                #just output 0, that is the majority.
                b_preds = [[0 for val in item] for item in b_logits]    
            else:
                b_preds = [[int(val>=0) for val in item] for item in b_logits] # >= 0 means prob >= 0.5
            
            preds += b_preds
            trues += b_trues
        
        assert len(preds) == len(trues), len(preds)
        assert len(preds[0]) == len(trues[0]) == self.args.num_labels, len(preds[0])

        results = self.get_results(preds, trues, loss_sum, total)

        if(self.args.store_preds):
            results["preds"] = preds
            results["trues"] = trues
        
        #Set back to train mode
        self.model.train()
        
        if(during_training):
            return results["mean_loss"], results["mean_positive_f1"]
        else:
            return results

    def get_results(self, preds, trues, loss_sum, total):

        results = {}

        results["mean_loss"] = loss_sum/total
        results["label_wise"] = [self.get_label_wise_results(preds, trues, i) for i in range(self.args.num_labels)]

        sum_acc, sum_positive_f1 = 0.0, 0.0

        for item in results["label_wise"]:
            sum_acc += item["accuracy"]
            sum_positive_f1 += item["positive_f1"]

        results["mean_accuracy"] = sum_acc/len(results["label_wise"])
        results["mean_positive_f1"] = sum_positive_f1/len(results["label_wise"])

        #Other Overall metrics
        results["UL-A"] = self.get_ula(preds, trues)
        results["Joint-A"] = self.get_joint_a(preds, trues)

        return results

    def get_ula(self, preds, trues):
        """
        utterance level accuracy: average fraction of labels which are predicted accurately in one utterance. 
        """
        assert len(preds) == len(trues)

        ulas = []
        for pred, true in zip(preds, trues):
            this_sum = 0.0
            for pi, ti in zip(pred, true):
                if(pi == ti):
                    this_sum += 1
            ulas.append(this_sum/len(pred))
        
        return np.mean(ulas)

    def get_joint_a(self, preds, trues):
        """
        joint accuracy: fraction of utterances in which all the labels are predicted correctly.
        """
        counts = 0.0

        for pred, true in zip(preds, trues):
            
            this_sum = 0
            for pi, ti in zip(pred, true):
                if(pi == ti):
                    this_sum += 1
            
            if(this_sum == len(pred)):
                counts += 1
        
        return counts/len(preds)

    def get_label_wise_results(self, preds, trues, label_ix):
        
        this_results = {}

        this_preds = [item[label_ix] for item in preds]
        this_trues = [item[label_ix] for item in trues]

        this_results["accuracy"] = accuracy_score(y_true=this_trues, y_pred=this_preds)
        this_results["positive_f1"] = f1_score(y_true=this_trues, y_pred=this_preds)
        this_results["cls_report"] = classification_report(y_true=this_trues, y_pred=this_preds)

        return this_results

    def get_ckpt(self):
        """
        create checkpoint object to be stored to a file
        """
        
        """
        Checkpoint the pytorch model.
        """
        ckpt = {}
        ckpt["ckpt"] = self.model.state_dict()
        ckpt["args"] = vars(self.args)
        ckpt["means"] = self.data.scalar.means
        ckpt["stds"] = self.data.scalar.stds
        
        return ckpt

    def load_model_from_ckpt(self, ckpt):
        """
        Primarily used to load the best checkpoint after training, so that final evaluation can be done in the same script. 
        """
        print("Inside load_model_from_ckpt ", file=self.logger)
        self.model.load_state_dict(ckpt["ckpt"])
