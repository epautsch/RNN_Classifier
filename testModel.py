from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# load the model from file
model = load_model('my_model.h5')


# example passage to classify
new_passage = "I thought it was good"

top_words = 5000
max_review_length = 500


# preprocess the passage
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(new_passage)

Sequence = tokenizer.texts_to_sequences(new_passage)
padded_sequence = sequence.pad_sequences(Sequence, maxlen=max_review_length)

# make prediction
prediction = model.predict(padded_sequence)

print(sum(prediction[0])/len(prediction[0]))

if sum(prediction[0])/len(prediction[0]) > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")