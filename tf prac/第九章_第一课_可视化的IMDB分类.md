```python
# NOTE: PLEASE MAKE SURE YOU ARE RUNNING THIS IN A PYTHON3 ENVIRONMENT

import tensorflow as tf
print(tf.__version__)

# This is needed for the iterator over the data
# But not necessary if you have TF 2.0 installed
#!pip install tensorflow==2.0.0-beta0


#tf.enable_eager_execution()

#!pip install -q tensorflow-datasets
```

    2.2.0



```python
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

```


```python
import numpy as np

train_data, test_data = imdb['train'], imdb['test']


training_sentences = []
training_labels = []

testing_sentences = []  
testing_labels = []


# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())
  
  
for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())
 
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

```


```python
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type, padding='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


```


```python
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])
```

    b'i have been known to fall asleep during films but this is usually due to a combination of things including really tired being warm and comfortable on the <OOV> and having just eaten a lot however on this occasion i fell asleep because the film was rubbish the plot development was constant constantly slow and boring things seemed to happen but with no explanation of what was causing them or why i admit i may have missed part of the film but i watched the majority of it and everything just seemed to happen of its own <OOV> without any real concern for anything else i cant recommend this film at all ' ? ? ? ? ? ? ?
    b'I have been known to fall asleep during films, but this is usually due to a combination of things including, really tired, being warm and comfortable on the sette and having just eaten a lot. However on this occasion I fell asleep because the film was rubbish. The plot development was constant. Constantly slow and boring. Things seemed to happen, but with no explanation of what was causing them or why. I admit, I may have missed part of the film, but i watched the majority of it and everything just seemed to happen of its own accord without any real concern for anything else. I cant recommend this film at all.'



```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 120, 16)           160000    
    _________________________________________________________________
    flatten (Flatten)            (None, 1920)              0         
    _________________________________________________________________
    dense (Dense)                (None, 6)                 11526     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 7         
    =================================================================
    Total params: 171,533
    Trainable params: 171,533
    Non-trainable params: 0
    _________________________________________________________________



```python
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
```

    Epoch 1/10
    782/782 [==============================] - 8s 10ms/step - loss: 0.4900 - accuracy: 0.7485 - val_loss: 0.3633 - val_accuracy: 0.8396
    Epoch 2/10
    782/782 [==============================] - 7s 9ms/step - loss: 0.2419 - accuracy: 0.9059 - val_loss: 0.3902 - val_accuracy: 0.8277
    Epoch 3/10
    782/782 [==============================] - 8s 10ms/step - loss: 0.0983 - accuracy: 0.9735 - val_loss: 0.4587 - val_accuracy: 0.8253
    Epoch 4/10
    782/782 [==============================] - 7s 9ms/step - loss: 0.0260 - accuracy: 0.9965 - val_loss: 0.5626 - val_accuracy: 0.8109
    Epoch 5/10
    782/782 [==============================] - 7s 9ms/step - loss: 0.0075 - accuracy: 0.9994 - val_loss: 0.6271 - val_accuracy: 0.8155
    Epoch 6/10
    782/782 [==============================] - 8s 10ms/step - loss: 0.0024 - accuracy: 0.9999 - val_loss: 0.6957 - val_accuracy: 0.8129
    Epoch 7/10
    782/782 [==============================] - 8s 10ms/step - loss: 9.2728e-04 - accuracy: 1.0000 - val_loss: 0.7510 - val_accuracy: 0.8116
    Epoch 8/10
    782/782 [==============================] - 6s 8ms/step - loss: 4.9620e-04 - accuracy: 1.0000 - val_loss: 0.7882 - val_accuracy: 0.8135
    Epoch 9/10
    782/782 [==============================] - 7s 9ms/step - loss: 2.8690e-04 - accuracy: 1.0000 - val_loss: 0.8439 - val_accuracy: 0.8108
    Epoch 10/10
    782/782 [==============================] - 7s 10ms/step - loss: 1.7213e-04 - accuracy: 1.0000 - val_loss: 0.9017 - val_accuracy: 0.8085





    <tensorflow.python.keras.callbacks.History at 0x7f7d6bff60d0>



这里可能存在过拟合的情况，因为训练集的准确程度升为1.


```python
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
```

    (10000, 16)



```python
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```


```python


# try:
#   from google.colab import files
# except ImportError:
#   pass
# else:
#   files.download('vecs.tsv')
#   files.download('meta.tsv')
```
