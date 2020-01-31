import os
import string
from fire import Fire

def load_doc(filename):
	with open(filename, 'r') as f_in:
		text = f_in.read()
	return text


def save_doc(lines, filename):
	data = '\n'.join(lines)
	with open(filename, 'w') as f_out:
		f_out.write(data)


def preprocess(file_name, out_filename): 
	raw_text = load_doc(file_name)
	tokens = raw_text.split()
	raw_text = ' '.join(tokens).lower()
	vocab_str = ' ' + string.ascii_letters + ''.join([str(x) for x in range(10)]) + string.punctuation
	raw_text = ''.join([x for x in raw_text if x in vocab_str])

	length = 10
	sequences = list()
	for i in range(length, len(raw_text)):
		seq = raw_text[i-length:i+1]
		sequences.append(seq)

	print(sequences)
	save_doc(sequences, out_filename)


if __name__ == '__main__': 
	Fire(preprocess)
