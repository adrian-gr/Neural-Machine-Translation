import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
# device = torch.device("cpu")

embedding_size = 1024
hidden_size = 256
num_layers = 4
dropout = 0.2
attention_size = 256

beta = 2

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 20
maxEpochs = 5
log_every = 10
test_every = 1000

max_patience = 5
max_trials = 5