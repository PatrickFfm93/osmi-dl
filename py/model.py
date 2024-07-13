import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from tensorflow.python.keras.layers.core import Activation
import tensorflowjs as tfjs

# Laden und Vorbereiten der Daten
texts = text = open('data/data.txt', encoding='utf-8').read().lower() 

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

# Modellarchitektur
model = Sequential()
model.add(Embedding(vocab_size, word_count, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax')) # len(tokenizer.word_index) + 1, activation='softmax'
model.summary()

# Kompilieren und Trainieren des Modells
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=32)

# Speichern des Modells
model.save('models/model.h5')
tfjs.converters.save_keras_model(model, 'models/')