```python

import os

# Directory with our training horse pictures
train_horse_dir = os.path.join('/home/jovyan/tensorflow_datasets/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/home/jovyan/tensorflow_datasets/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/home/jovyan/tensorflow_datasets/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/home/jovyan/tensorflow_datasets/validation-horse-or-human/humans')
```

## Building a Small Model from Scratch

But before we continue, let's start defining the model:

Step 1 will be to import tensorflow.


```python
import tensorflow as tf
```

We then add convolutional layers as in the previous example, and flatten the final result to feed into the densely connected layers.

Finally we add the densely connected layers. 

Note that because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).


```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```


```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])
```


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/home/jovyan/tensorflow_datasets/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/home/jovyan/tensorflow_datasets/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

    Found 1027 images belonging to 2 classes.
    Found 256 images belonging to 2 classes.



```python
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=5,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
```

    WARNING:tensorflow:From <ipython-input-6-726b7db7246c>:7: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.fit, which supports generators.
    Epoch 1/5
    8/8 [==============================] - 75s 9s/step - loss: 0.7182 - acc: 0.5028 - val_loss: 0.6880 - val_acc: 0.7305
    Epoch 2/5
    8/8 [==============================] - 72s 9s/step - loss: 0.6700 - acc: 0.5929 - val_loss: 0.6857 - val_acc: 0.5742
    Epoch 3/5
    8/8 [==============================] - 71s 9s/step - loss: 0.6459 - acc: 0.6541 - val_loss: 0.6826 - val_acc: 0.5430
    Epoch 4/5
    8/8 [==============================] - 72s 9s/step - loss: 0.6419 - acc: 0.6618 - val_loss: 0.6847 - val_acc: 0.5000
    Epoch 5/5
    8/8 [==============================] - 83s 10s/step - loss: 0.5934 - acc: 0.7141 - val_loss: 0.6851 - val_acc: 0.5000



```python
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)


输出结果也是满奇怪的，validation loss几乎没怎么变过
