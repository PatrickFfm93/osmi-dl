import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import json
from tensorflow.python.keras.layers.core import Activation
import tensorflowjs as tfjs

path = 'data/data.txt'
text = open(path, encoding='utf-8').read().lower()

def clean_text(doc):
    tokens = doc.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    return tokens

tokens = clean_text(text)
word_count = 5
word_count = word_count + 1
lines = []

for i in range(word_count, len(tokens)):
    seq = tokens[i-word_count:i]
    line = ' '.join(seq)
    lines.append(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

X = tf.constant([item[:-1] for item in sequences])
y = tf.constant([item[-1] for item in sequences])

vocab_size = len(tokenizer.word_index) + 1

y = to_categorical(y, num_classes=vocab_size)

seq_length = word_count - 1


with open('tokenizer_word_index.json', 'w', encoding='utf-8') as file:
    json.dump(tokenizer.word_index, file, ensure_ascii=False)

with open('tokenizer_index_word.json', 'w', encoding='utf-8') as file:
    json.dump(tokenizer.index_word, file, ensure_ascii=False)

model = Sequential()
model.add(Embedding(vocab_size, word_count, input_length=seq_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=400)

model.save('models/next_word_model.h5')
tfjs.converters.save_keras_model(model, 'models')