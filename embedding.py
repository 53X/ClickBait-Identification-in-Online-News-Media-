'''
Necessary Library Imports
'''

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from get_data import get_data
import numpy as np
import gensim
from gensim.models import KeyedVectors

def preprocess():

	texts, annotations = get_data()

	'''

	Doing preprocessing stuffs on my text data--- Tokenizing, 
												  Generating word sequences,
								    			  Padding sequences upto a maximum length
	'''

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	padded = pad_sequences(sequences, maxlen=200, padding='post')

	word_index = tokenizer.word_index
	print("Length of Word Index found in corpus: {} ".format(len(word_index)))

	return sequences, annotations, word_index


def get_embedding(word_dimension, embedding='glove840'):

	embedding_matrix = np.zeros(shape=(len(word_index)+1, word_dimension), dtype='float32')

	word_index = preprocess()[-1]

	'''
	Loading our standard model 

	'''

	if embedding == 'glove840':
		vocab_model = Keyedvectors.load_word2vec_format('glove_840B_300d.txt',binary=False)
	elif embedding == 'glove42':
		vocab_model = KeyedVectors.load_word2vec_format('glove_42B_300d.txt', binary = False)
	else:
		vocab_model = KeyedVectors.load_word2vec_format('mikolov_word2vec.bin', binary=True)







