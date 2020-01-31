import numpy
from numpy import array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Dense,Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
import pickle
import warnings
import os
import json
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")

def load_doc(filename):
	with open(filename, 'r') as f_in:
		text = f_in.read()
	return text

def get_chars_mapping_input_lines(input_file):
	raw_text = load_doc(in_filename)
	lines = raw_text.split('\n')
	chars = sorted(list(set(raw_text)))
	mapping = dict((c, i) for i, c in enumerate(chars))
	return mapping, chars, lines

def get_sequences(mappper, lines):
	sequences = list()
	for line in lines:
		encoded_seq = [mappper[char] for char in line]
		sequences.append(encoded_seq)
	return sequences

def train_test_split(sequences, train_pct = 1, shuffle: bool = True):
	sequences = array(sequences)
	if shuffle:
		numpy.random.shuffle(sequences)
	train_sequences, test_sequences = sequences[:int(len(sequences)*train_pct)], sequences[int(len(sequences)*train_pct):]
	X_train, y_train = train_sequences[:,:-1], train_sequences[:,-1]
	X_test, y_test = test_sequences[:,:-1], test_sequences[:,-1]
	return X_train, y_train, X_test, y_test

def one_hot_encoding(X, y, no_of_cat):
	X = array([to_categorical(x, num_classes=no_of_cat) for x in X])
	y = to_categorical(y, num_classes=no_of_cat)
	return X, y

def build_network(params, model_name):
	model = Sequential()
	model.add(LSTM(2 ** 9, return_sequences=True, input_shape=(params['sequence_lenght'], params['vocabulary_size'])))
	model.add(Dropout(0.4))
	model.add(LSTM(2 ** 9, return_sequences=True))
	model.add(Dropout(0.4))
	model.add(Bidirectional(LSTM(2 ** 9, return_sequences=True)))
	model.add(Dropout(0.4))
	model.add(LSTM(2 ** 9, return_sequences=False))
	model.add(Dropout(0.4))
	model.add(Dense(params['vocabulary_size'], activation='softmax'))
	model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params['metrics'])
	if not os.path.isdir(os.path.join('data', model_name)):
		os.makedirs(os.path.join('data', model_name))
	with open(os.path.join('data', model_name, 'model_summary.txt'), 'w') as f:
		with redirect_stdout(f):
			print(model.summary())
	return model

def save_model(results, params, name):
	if not os.path.isdir(os.path.join('data', name)):
		os.makedirs(os.path.join('data', name))
	pickle.dump(mapping, open(os.path.join('data', name, 'mapping.pkl'), 'wb'))
	with open(os.path.join('data', name, 'training_history.json'), 'w') as json_out:
		json.dumps(str(results.history),indent=2)
	with open(os.path.join('data', name, 'params.json'), 'w') as json_out:
		json.dumps(str(params), indent=2)


# Parameters
hyperp = {}
# in_filename = os.path.join('data', 'char_sequences.txt')
in_filename = 'titles_processed.txt'
model_name = 'titles_bi_7_length'
hyperp['loss'] = 'categorical_crossentropy'
hyperp['optimizer'] = 'adam'
hyperp['metrics'] = ['accuracy']
hyperp['training_percentage'] = .8
assert 0 < hyperp['training_percentage'] < 1, 'Training percentage must be value between 0 and 1'
hyperp['epochs'] = 300
hyperp['batch_size'] = 1024
hyperp['random_seed'] = 1
hyperp['patience'] = 10

# Execution
numpy.random.seed(hyperp['random_seed'])
mapping, chars, lines = get_chars_mapping_input_lines(in_filename)
vocab_size = len(mapping)
hyperp['vocabulary_size'] = vocab_size
seqs = get_sequences(mapping, lines)
hyperp['sequence_lenght'] = len(seqs[0]) - 1

X_train, y_train, X_test, y_test = train_test_split(seqs, hyperp['training_percentage'])
X_train, y_train = one_hot_encoding(X_train, y_train, vocab_size)
X_test, y_test = one_hot_encoding(X_test, y_test, vocab_size)

char_rnn = build_network(hyperp, model_name)
early_stopper = EarlyStopping(patience=hyperp['patience'], verbose=1)
check_point_path = os.path.join(os.path.join('data', model_name, 'model.h5'))
check_pointer = ModelCheckpoint(check_point_path, verbose=1, save_best_only=True)
results = char_rnn.fit(X_train, y_train, epochs=hyperp['epochs'], verbose=1, batch_size=hyperp['batch_size'],
	validation_data=[X_test, y_test], callbacks=[early_stopper, check_pointer])

save_model(results, hyperp, model_name)
