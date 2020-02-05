#!/usr/bin/env python
# coding: utf-8

import string
import numpy as np
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)




def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','', string.punctuation))




def add2dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict




initial_word = {}
second_word = {}
transitions = {}





# Trains a Markov model based on the data in training_data_file
def train_markov_model():
    training_data_file = open('../../data/titles.txt').read().splitlines()
    for line in training_data_file:
        tokens = remove_punctuation(line.rstrip().lower()).split()
        tokens_length = len(tokens)
        for i in range(tokens_length):
            token = tokens[i]
            if i == 0:
                initial_word[token] = initial_word.get(token, 0) + 1
            else:
                prev_token = tokens[i - 1]
                if i == tokens_length - 1:
                    add2dict(transitions, (prev_token, token), 'END')
                if i == 1:
                    add2dict(second_word, prev_token, token)
                else:
                    prev_prev_token = tokens[i - 2]
                    add2dict(transitions, (prev_prev_token, prev_token), token)
    
    initial_word_total = sum(initial_word.values())
    for key, value in initial_word.items():
        initial_word[key] = value / initial_word_total
        
    for prev_word, next_word_list in second_word.items():
        second_word[prev_word] = list2probabilitydict(next_word_list)
        
    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = list2probabilitydict(next_word_list)
    
    print('Training successful.')




train_markov_model()


def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key
    assert(False)





number_of_sentences = 10




def generate():
    for i in range(number_of_sentences):
        sentence = []
        word0 = sample_word(initial_word)
        sentence.append(word0)
        word1 = sample_word(second_word[word0])
        sentence.append(word1)
        # Subsequent words untill END
        while True:
            word2 = sample_word(transitions[(word0, word1)])
            if word2 == 'END':
                break
            sentence.append(word2)
            word0 = word1
            word1 = word2
        print(' '.join(sentence))



generate()

