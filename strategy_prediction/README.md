This directory contains our code for the multi-tasking framework for strategy prediction from a given utterance. Please refer to the paper for more details.

# File Descriptions
**main.py**: main driver code\
**arguments.py**: All the hyperparameters and i/o arguments -> controlled from main.py\
**data.py**: handles the dataloaders\
**model.py**: handles the model architecture, training, and evaluation

# Notes

* Path names in CAMEL_CASE format must be replaced by proper paths before running the code.
* At this point, the code assumes a specific input format that may not directly map to the provided dataset files in this repository. Till this is fixed, the hope is that this mapping is easy to do. If you face any issues, please feel free to get in touch.
