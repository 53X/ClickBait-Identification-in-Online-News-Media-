import keras
from keras.layers import Embedding, Dense, Input, GRU, Bidirectional,  Dropout
from keras.models import Model 
from embedding import preprocess, get_embedding
from keras.optimizers import RMSprop



vocabulary = len(preprocess()[-1]) + 1
optimizer = RMSprop()

def clickbait_model(maxlen=200, embedding_type='glove840'):

	input_layer = Input(shape=(maxlen, ), name='input_layer')

	input_drop = Dropout(0.2)(input_layer)

	'''
	Dropout for the embedding layer

	'''

	embedding = Embedding(input_dim=vocabulary, output_dim=300, input_length=maxlen,name='embedding_layer')(input_drop)

	'''
	Dropout for the input of the Bidirectional GRU

	''' 

	embedding_drop = Dropout(0.2)(embedding)
	
	
	'''
	
	This is the GRU layer that will be used for capturing the sequential dependencies
	in our model both along the forward and backward directions.
	
	'''

	my_gru = GRU(128, input_shape=(maxlen, 300),  return_sequences=False)

	'''
	This layer uses the bidirectional wrapper around our custom GRU to model
	long term dependencies in the sequence along both forward and backward directions.
	The merge_mode parameter indicates that the final output will be a concatenation
	of the state-vectors along both forward and backward directions. Thus for our case
	the output of our 'bidirectional' layer will be a single vector of 256 dimensions.

	'''

	bidirectional = Bidirectional(my_gru, merge_mode='concat')(embedding_drop)

	'''
	Dropout for the output of the Bidirectional GRU 

	'''
	
	bidirectional_drop = Dropout(0.5)(bidirectional)

	output = Dense(1, activation='sigmoid', name='output_layer')(bidirectional_drop)

	model = Model(inputs=input_layer, outputs=output)

	model.compile(optimizer, loss='binary_crossentropy',  metrics=['accuracy','mse'])

	print("Model Ready for Deployment:")

	print(model.summary())

	return model



if __name__ =='__main__':

	clickbait_model()



