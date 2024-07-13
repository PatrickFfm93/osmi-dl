from keras.models import load_model
import json
import tensorflow as tf
import os


model = load_model('models/next_word_model.h5')

tokenizer_word = json.loads(open('tokenizer_word_index.json', encoding='utf-8').read())
tokenizer_index = json.loads(open('tokenizer_index_word.json', encoding='utf-8').read())

input_text = "wir laufen zusammen durch den"
text = input_text.split()[-5:]
encoded = []
for word in text:
    encoded.append(tokenizer_word[word])

encoded = tf.constant([encoded])
pred = model.predict(encoded)[0].tolist()
pred_class = model.predict_classes(encoded)
temp = {}
for i, c in enumerate(pred):
    temp[i] = c
sort_temp = sorted(temp.items(), key=lambda x:x[1], reverse=True)

os.system('cls')
print(input_text + ' ...')
for i in range(5):
    print(tokenizer_index[str(sort_temp[i][0])] + ': ' + str(sort_temp[i][1]))

# print(tokenizer_index[str(pred[0])])