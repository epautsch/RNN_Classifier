import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import pandas
import numpy as np


# fix random seed for reproducibility
tf.random.set_seed(7)

# Load the data
data = pandas.read_csv('training.csv', usecols=['text', 'author'], sep=',', encoding='ISO-8859-1')
texts = data['text']  # Extract the text column
labels = data['author']-1  # Extract the label column

# Convert the text into numerical sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


# Pad the sequences to a fixed length
max_sequence_length = 1000  # Set the maximum sequence length
sequences = sequence.pad_sequences(sequences, maxlen=max_sequence_length)


# Encode the labels into one-hot vectors
num_classes = 50  # Set the number of classes
labels = to_categorical(labels, num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)


# load the dataset but only keep the top n words, zero the rest
top_words = 10001


# truncate and pad input sequences
'''
max_review_length = 1000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
'''

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_sequence_length))

model.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))
model.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))

model.add(Dense(50, activation='softmax')) # Output layer with 50 units and softmax activation

def cosine_annealing(epoch, initial_learning_rate, T_max):
    return initial_learning_rate * (1 + np.cos(np.pi * epoch / T_max)) / 2

initial_learning_rate = 0.001
T_max = 50  # Total number of epochs

lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, initial_learning_rate, T_max))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Use categorical_crossentropy loss for multiclass classification
print(model.summary())

filepath="best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32, callbacks=[checkpoint])


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# model.save('authorship_rnn_model.h5')

