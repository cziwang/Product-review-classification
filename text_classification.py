import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

word_index = {key: (value + 3) for key, value in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK'] = 2
word_index['<UNUSED'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])

# Each review varies in length, so we will add '<PAD>'
# to shorter reviews. This way, all reviews will be
# the same length.
# Same length is required for the input layer in NN

# Preprocess data to maxlength of 256
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=250)

print(len(train_data[0]))

# Create model
model = keras.Sequential()
model.add(keras.layers.Embedding(80000, 16))    # turns each individual input into 16 dimensional vectors
model.add(keras.layers.GlobalAveragePooling1D())    # averages vectors in the embedding layer
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save('model.h5')

# To reload model after saving,
# model = keras.models.load_model('model.h5')

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

with open('harry-potter-review.txt', encoding='utf-8')  as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences(
            [encode], value=word_index["<PAD>"], padding="post", maxlen=250
        ) # make data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

test_review = test_data[0]
predict = model.predict([test_review])
print('Review:', decode_review(test_review))
print('Prediction:', predict)
print('Actual:', test_labels[0])
print(results)
