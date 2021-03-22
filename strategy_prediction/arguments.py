"""
Class to parse and validate command-line arguments required by train, classify and evaluate.
"""
class ArgumentHandler:

    def __init__(self):

        #I/O
        
        #data.py: DataHandler
        self.logdir = None

        self.max_tokens = 200
        self.labels = ""
        self.num_labels = 3
        self.feature_input_dims = self.num_labels + 1 #turn no + context bow
        self.context_size = 3

        #model.py: Net
        self.text_encoder = True
        self.text_encoder_dims = 768
        self.text_output_dims = 64
        self.bert = True

        self.feature_encoder = False
        self.feature_encoder_dims = 64

        self.dropout = 0.1
        self.activation = 'relu'

        #model.py: ModelHandler
        self.batch_size = 8
        self.load_from_ckpt = False
        self.ckpt_file = None #"best_ckpt.pt"

        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        self.num_epochs = 500 #only using num_iters, essentially.
        self.num_iters=100
        self.print_iter_no_after = 25#print iter number after.
        self.ckpt_after = 50

        self.freeze = False

        self.oversample = False
        self.have_attn = True

        self.pretraining = False
        self.pretrained_dir = ""

        self.store_preds = False
        
        self.majority_prediction = False

    def update_hyps(self, hyps):
        """
        Update the params
        """
        for key, value in hyps.items():
            setattr(self, key, value)