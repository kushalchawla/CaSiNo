import os, sys
import random
import torch
import json
import pandas as pd
from matplotlib.pylab import plt
from sklearn.model_selection import KFold

from arguments import ArgumentHandler
from data import DataHandler
from model import ModelHandler
#from torch.utils.tensorboard import SummaryWriter
#from pytorch_model_summary import summary

"""
For every hyp combination:
    For every cv split
        Basically: lets do this, restrict this to just one round of CV, somehow.
"""

def select_hyps(hyp_search):
    exceptions = set(['labels'])
    this_hyp = {}
    
    for hyp, val in hyp_search.items():
        
        if isinstance(val, list) and (hyp not in exceptions):
            this_hyp[hyp] = random.choice(val)
        else:
            this_hyp[hyp] = val
            
    return this_hyp

def get_dialogs_from_file(fname):
    
    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
    
    df = pd.read_csv(fname)
    dat = pd.DataFrame.to_dict(df, orient="records")
    
    all_data = []
    
    cur_dialog = []
    cur_id = -1
    
    for i, item in enumerate(dat):
        
        if((item["DialogueId"] != item["DialogueId"]) or (item["Utterance"] in extra_utterances)):
            continue
            
        #valid item
        this_id = item["DialogueId"]
        if((this_id != cur_id) and cur_dialog):
            #found new id
            new_dialog = cur_dialog[:]
            all_data.append(new_dialog)
            cur_dialog = []
            
        if(isinstance(item["Labels"], str)):
            assert isinstance(item["Labels"], str) and len(item["Labels"])>0, i
            this_item = [item["Utterance"], item["Labels"]]
            cur_dialog.append(this_item)
            cur_id = this_id
            
    if(cur_dialog):
        new_dialog = cur_dialog[:]
        all_data.append(new_dialog)
        cur_dialog = []
        
    return all_data

def get_all_data(input_files):
    """
    list of dialogues with annotations.
    """
    all_data = []
    
    for input_file in input_files:
        all_data += get_dialogs_from_file(input_file)
    
    return all_data

def get_all_data_dummy():
    """
    simple list of dialogues with annotations. To illustrate the expected input format.
    """
    """
    #DUMMY FOR NOW
    all_data = [
            
        [
            ["Hello!  Let's work together on a deal for these packages, shall we? What are you most interested in?",
            "Promote-coordination,Preference-elicitation"],
            ["Hey! I'd like some more firewood to keep my doggo warm. What do you need?",
            "Required-other,Preference-elicitation"],
        ],
        [
            ["Hello!  Let's work together on a deal for these packages, shall we? What are you most interested in?",
            "Promote-coordination,Preference-elicitation"],
            ["Hey! I'd like some more firewood to keep my doggo warm. What do you need?",
            "Required-other,Preference-elicitation"],
        ],
    ]

    all_data = all_data*100
    
    return all_data
    """
    pass

def get_cv_generator(all_data, num_folds):
    kf = KFold(n_splits=num_folds)
    train_ratio = 0.95
    X = [[0] for _ in range(len(all_data))]
    for train_index, test_index in kf.split(X):
        train_dev_data = [all_data[ii] for ii in train_index]
        test_data = [all_data[ii] for ii in test_index]
        ll = len(train_dev_data)
        train_data, dev_data = train_dev_data[:int(ll*train_ratio)], train_dev_data[int(ll*train_ratio):]
        yield (train_data, dev_data, test_data)
        
def write_graph(net, device, dataloader):
    inputs = None
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, _ = batch
        inputs = (x_input_ids, x_input_type_ids, x_input_mask, x_input_feats)
        break
    
    board_dir = BOARD_DIR
    writer = SummaryWriter(board_dir)
    writer.add_graph(net, inputs)
    writer.close()
    print("GRAPH WRITTEN FOR TENSORBOARD AT: ", board_dir, file=logger)

def show_summary_func(net, device, dataloader):
    x_input_ids, x_input_type_ids, x_input_mask, x_input_feats = None, None, None, None
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, _ = batch
        break
    
    print(summary(net, x_input_ids, x_input_type_ids, x_input_mask, x_input_feats, show_input=True), file=logger)

def train_and_eval(cv_data, hyps, logger, show_summary=False):
    train_data, val_data, test_data = cv_data[0], cv_data[1], cv_data[2]
    
    #arguments
    args = ArgumentHandler()
    args.update_hyps(hyps)
    
    #flush the output
    logger.flush()

    #data
    data = DataHandler(args=args, logger=logger)
    data.train_dataloader = data.get_dataloader(train_data, fit_scalar=True, train=True)
    data.dev_dataloader = data.get_dataloader(val_data)
    data.test_dataloader = data.get_dataloader(test_data)
    
    #flush the output
    logger.flush()
    
    #modelling
    model = ModelHandler(data=data, args=args, logger=logger)
    
    if(show_summary):
        show_summary_func(model.model, model.device, data.train_dataloader)
    
    best_dev_f1, best_ckpt, summary = model.train_model()
    
    #now evaluate on the best_ckpt
    model.load_model_from_ckpt(best_ckpt)#load_best_ckpt basically.
    #get results
    train_results = model.evaluate_model("train", sample_size=-1)
    dev_results = model.evaluate_model("dev", sample_size=-1)
    test_results = model.evaluate_model("test", sample_size=-1)
    
    #build output dict for this run:
    output = {}
    
    output["model"] = best_ckpt
    output["results"] = {
        "train": train_results,
        "dev": dev_results,
        "test": test_results,
    }
    output["training"] = {
        "ckpt_level_summary": summary,
        "best_dev_f1": best_dev_f1
    }
    
    del args
    del data
    del model
    
    return output
    
def get_cv_mean_f1_on_val(cv_results):
    
    sum_f1 = 0.0
    for item in cv_results:
        sum_f1 += item["dev"]["mean_positive_f1"]
    
    return sum_f1/len(cv_results)

def merge_cv_results(cv_results):
    """
    Means across CV
    """
    dtypes = ["train", "dev", "test"]
    props_l1 = ["mean_loss", "mean_accuracy", "mean_positive_f1", "UL-A", "Joint-A"]
    props_l2 = ["accuracy", "positive_f1"]
    
    merged_results = {}
    
    for dtype in dtypes:
        merged_results[dtype] = {}
        for prop in props_l1:
            summ = 0.0
            for item in cv_results:
                summ += item[dtype][prop]
            merged_results[dtype][prop] = summ/len(cv_results)
            
        num_labels = len(cv_results[0][dtype]["label_wise"])
        merged_results[dtype]["label_wise"] = [{} for _ in range(num_labels)]
        for i in range(num_labels):
            for prop in props_l2:
                summ = 0.0
                for item in cv_results:
                    summ += item[dtype]["label_wise"][i][prop]
                merged_results[dtype]["label_wise"][i][prop] = summ/len(cv_results)
    
    return merged_results

def save_everything(all_cv_outputs, ckptfile, cv_wise_file, hyp_wise_file):
    """
    models and results for the cvs, individually and aggregate data.
    """
    #models
    models = all_cv_outputs["models"]
    torch.save(models, ckptfile)
    
    #cv wise
    cv_wise = {
        "results": all_cv_outputs["results"],
        "training": all_cv_outputs["training"],
    }
    with open(cv_wise_file, 'w') as fp:
        json.dump(cv_wise, fp)
    
    #hyp level
    hyp_wise = {
        "hyps": all_cv_outputs["hyps"],
        "results": merge_cv_results(all_cv_outputs["results"])
    }
    with open(hyp_wise_file, 'w') as fp:
        json.dump(hyp_wise, fp)
        
input_files = [
    PATH TO ANNOTATION FILES
]

all_data = get_all_data(input_files)

print("Total number of dialogues in the dataset: ", len(all_data))

total_utts = 0
for item in all_data:
    total_utts += len(item)
print("Total utterances: ", total_utts)

logdir = LOG_DIR
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
ckptfile = os.path.join(logdir, "best_ckpt.pt")
cv_wise_file = os.path.join(logdir, "cv_wise.json")
hyp_wise_file = os.path.join(logdir, "hyp_wise.json")
logger = open(os.path.join(logdir, "all_logs.txt"), "w")


labels = ['Small-talk', 'Required-self', 'Required-other', 'Not-Required', 'Preference-elicitation',
         'Undervalue-Other-Requirement', 'Vouching-for-fairness']

num_labels = len(labels)
feature_input_dims = 32 #len(labels) + 32 #position embedding is 32.

batch_size = 64
ckpt_after = (4615//batch_size)
num_iters = 10*ckpt_after

print("batch_size, ckpt_after, num_iters: ", batch_size, ckpt_after, num_iters)

hyp_search_trials = 1
hyp_search = {
    'logdir': logdir,
    'labels': labels,
    'num_labels': num_labels,
    'feature_input_dims': feature_input_dims,

    'text_encoder': True,
    'bert': True,
    'text_encoder_dims': 768,
    'text_output_dims': [128],

    'feature_encoder': True,
    'feature_encoder_dims': [128],

    'batch_size': [batch_size],
    'num_iters': num_iters,
    'ckpt_after': ckpt_after,

    'freeze': False,
    'oversample': True,
    'have_attn': True,
    'store_preds': False,
    'majority_prediction': False,

    'pretraining': False,
    'pretrained_dir': PATH_TO_PRETRAINED_MODEL,

    'print_iter_no_after': 25,
    'activation': ['relu'],
    'learning_rate': [5e-5],
    'weight_decay': [0.01],
    'dropout': [0.1],
}

best_hyps = None
best_f1 = -1 #best avg f1 score for positive class for a given hyp -> avg over label set and CV split.
best_outputs = None #output results for the best model configuration.
all_f1 = [] #all avg f1 scores for positive classs. -> avg over label set and CV split.

#CV
num_cv_folds_total = 5
num_cv_folds_used = 5

print("SEARCH STARTING: hyp_search_trials, num_cv_folds_total, num_cv_folds_used", hyp_search_trials, num_cv_folds_total, num_cv_folds_used, file=logger)
for trial in range(hyp_search_trials):
    print("-------------------------------------------------", file=logger)
    this_hyps = select_hyps(hyp_search)
    #get CV data generator
    cv_gen = get_cv_generator(all_data, num_cv_folds_total)
    all_cv_outputs = {
        "results": [],
        "models": [],
        "training": [],
        "hyps": this_hyps
    } #results+models
    
    for cv_no, cv_data in enumerate(cv_gen):
        if(cv_no >= num_cv_folds_used):
            #for testing purposes.
            break
        
        cv_output = train_and_eval(cv_data, this_hyps, logger)
        print("cv_no, mean_positive_f1 for dev: ", cv_no, cv_output["results"]["dev"]["mean_positive_f1"], file=logger)
        #store
        all_cv_outputs["results"].append(cv_output["results"])
        #all_cv_outputs["models"].append(cv_output["model"])#save models
        all_cv_outputs["training"].append(cv_output["training"])
        
        #flush the output
        logger.flush()
        torch.cuda.empty_cache() # PyTorch thing
        
    hyp_f1 = get_cv_mean_f1_on_val(all_cv_outputs["results"])
    
    if(hyp_f1 > best_f1):
        #store the newer models instead, save hyps
        best_f1 = hyp_f1
        best_hyps = this_hyps
        
        #so we will save the models and the results, analysis, stats for those models.
        save_everything(all_cv_outputs, ckptfile, cv_wise_file, hyp_wise_file)
        print("Everything stored!!!", file=logger)
        print("ckptfile, cv_wise_file, hyp_wise_file", ckptfile, cv_wise_file, hyp_wise_file, file=logger)
    
    #for hyp level stats.
    all_f1.append(hyp_f1)
    print("Trial number, best_f1, hyp_f1, this_hyps: ", trial, best_f1, hyp_f1, this_hyps, file=logger)
    
    #flush the output
    logger.flush()
    
    torch.cuda.empty_cache() # PyTorch thing
    
print("SEARCH FINISHED", file=logger)
print("Overall best_f1, best_hyps: ", best_f1, best_hyps, file=logger)
print("Distribution of val F1s for all hyps: ", file=logger)
print(pd.Series(all_f1).describe(), file=logger)

logger.close()

