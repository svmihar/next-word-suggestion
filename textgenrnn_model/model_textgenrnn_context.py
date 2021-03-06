from textgenrnn import textgenrnn
from datetime import datetime
import os 

model_cfg = {
    'word_level': True,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 700,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 5,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': True,   # consider text both forwards and backward, can give a training boost
    'max_length': 7,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 100000,   # maximum number of words to model; the rest will be ignored (word-level model only)
}

train_cfg = {
    'line_delimited': True,   # set to True if each text has its own line in the source file
    'num_epochs': 700,   # set higher to train the model for longer
    'gen_epochs': 50,   # generates sample text from model after given number of epochs
    'train_size': 0.5,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.5,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': True   # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

file_name = '../data/title_with_le.csv'
model_name='v2_context'

textgen = textgenrnn(name=model_name)

train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

train_function(
    file_path=file_name,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=512,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level'], 
    context=True)

# this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
# changing the temperature schedule can result in wildly different output!
temperature = [1.0, 0.75, 0.5, 0.2]   
prefix = 'cuci ac'   # if you want each generated text to start with a given seed text

if train_cfg['line_delimited']:
  n = 1000
  max_gen_length = 60 if model_cfg['word_level'] else 300
else:
  n = 1
  max_gen_length = 2000 if model_cfg['word_level'] else 10000
  
timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)

textgen.generate_to_file(gen_file,
                         temperature=temperature,
                         prefix=prefix,
                         n=n,
                         max_gen_length=max_gen_length)
