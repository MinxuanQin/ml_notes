## 更复杂的图像应用：区分人和马

实验应用os库来进行文件操作

`
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import zipfile
`

### os&zipfile

|statements|explanation|
|----------|-----------|
|zip_ref = zipfile.ZipFile(local_zip, 'r')|创建ZipFile对象|
|zip_ref.extractall('/tmp/horse-or-human')|解压，参数为文件名|
|zip_ref.close()|关闭文件|
|train_horse_dir = os.path.join('/tmp/horse-or-human/**horses**')|将文件夹赋给train_horse_dir|
|train_horse_names = os.listdir(train_horse_dir)|得到存储所有文件夹下文件名的list|
|os.path.join()|拼接路径|
|os.getcwd()|得到当前工作目录名|


### matplotlib函数

|function|explanation|
|---------|----------|
|fig = plt.gcf()|get current figure-->便于对图片进行操作|
|fig.set_size_inches(ncols\*4, nrows\*4)|修改大小|
|mpimg.imread(img_path)|读取图片|
|for i,img_path in enumerate(next_horse_pix + next_human_pix)|**enumerate**形成一个字典，使i获得indice|
|for layer_name, feature_map in zip(layer_names, successive_feature_maps):|**zip**将列表中的一个个元素打包成元组

### 构建模型
本次采用RMSprop做优化器（和sgd相比可以自动调节学习率）。

#### 图像预处理 *`from tensorflow.keras.preprocessing.image import ImageDataGenerator`*
1. rescale
2. flow(data,labels) or flow_from_directory(directory, **tragetsize, batch_size, class_mode**)

**在此处为了最优化参数用pip install安装keras-tuner**


```python
%config IPCompleter.greedy = True
```


```python
#实际演示，已经将图片存储在以下路径
#/Users/qinminxuan/Documents/MLCourse/datasets/horse-or-human
import os
import zipfile

train_horse_dir = os.path.join('/Users/qinminxuan/Documents/MLCourse/datasets/horse-or-human/horses')
train_human_dir = os.path.join('/Users/qinminxuan/Documents/MLCourse/datasets/horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
```


```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
```


```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['acc'])
```


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255)

#batch 64，not 32！
train_generator = train_datagen.flow_from_directory(
    '/Users/qinminxuan/Documents/MLCourse/datasets/horse-or-human/',target_size = (150,150),batch_size = 64,
     class_mode = 'binary')
```

    Found 1027 images belonging to 2 classes.



```python
history = model.fit(train_generator, steps_per_epoch = 8, epochs = 15,verbose = 1)
```

    Epoch 1/15
    8/8 [==============================] - 3s 323ms/step - loss: 0.1871 - acc: 0.9335
    Epoch 2/15
    8/8 [==============================] - 3s 381ms/step - loss: 0.1491 - acc: 0.9434
    Epoch 3/15
    8/8 [==============================] - 4s 443ms/step - loss: 0.1161 - acc: 0.9579
    Epoch 4/15
    8/8 [==============================] - 3s 385ms/step - loss: 0.1197 - acc: 0.9590
    Epoch 5/15
    8/8 [==============================] - 3s 320ms/step - loss: 0.1358 - acc: 0.9424
    Epoch 6/15
    8/8 [==============================] - 3s 349ms/step - loss: 0.0599 - acc: 0.9845
    Epoch 7/15
    8/8 [==============================] - 3s 374ms/step - loss: 0.0803 - acc: 0.9688
    Epoch 8/15
    8/8 [==============================] - 3s 381ms/step - loss: 0.0800 - acc: 0.9746
    Epoch 9/15
    8/8 [==============================] - 3s 382ms/step - loss: 0.0532 - acc: 0.9863
    Epoch 10/15
    8/8 [==============================] - 3s 374ms/step - loss: 0.0536 - acc: 0.9805
    Epoch 11/15
    8/8 [==============================] - 3s 353ms/step - loss: 0.0716 - acc: 0.9756
    Epoch 12/15
    8/8 [==============================] - 3s 352ms/step - loss: 0.0448 - acc: 0.9867
    Epoch 13/15
    8/8 [==============================] - 3s 344ms/step - loss: 0.0465 - acc: 0.9845
    Epoch 14/15
    8/8 [==============================] - 3s 351ms/step - loss: 0.0405 - acc: 0.9889
    Epoch 15/15
    8/8 [==============================] - 3s 346ms/step - loss: 0.0328 - acc: 0.9889


参考对比不同时期做出的结果，感觉准确率有点波动是正常的，但是波动过大就要考虑参数的正确性，在这里之前是**RMSprop的lr过大**（0.01->0.001）

错误输出

![c4p2](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c4p2.jpg)


```python
from google.colab import files
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-19-ee201b91671e> in <module>
    ----> 1 from google.colab import files
    

    ModuleNotFoundError: No module named 'google.colab'



```python
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

#不太好的解决除std等于0的方法
np.seterr(divide='ignore',invalid='ignore')


successive_outputs = [layer.output for layer in model.layers[1:]]

visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

horse_img_files = [os.path.join(train_horse_dir,f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir,f) for f in train_human_names]
img_path = random.choice(horse_img_files+human_img_files)

img = load_img(img_path, target_size = (150,150))
x = img_to_array(img)
print(x.shape)

x = x.reshape((1,)+x.shape)
x = x/255

successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if(len(feature_map.shape) == 4):
        n_features = feature_map.shape[-1]
        #feature map has shape(None,size,size,n_features)
        size = feature_map.shape[1]
        
        display_grid = np.zeros((size,size*n_features))
        for i in range(n_features):
            x = feature_map[0,:,:,i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x,0,255).astype('uint8')
            
            display_grid[:,i*size:(i+1)*size] = x
            
        scale = 20./n_features
        plt.figure(figsize = (scale*n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
```

这里是因为网络太大，内存资源不够没能显示出来

图片如下：
![c4p1](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c4p1.png)


```python
validation_datagen = ImageDataGenerator(rescale = 1/255)

#batch 64，not 32！
validation_generator = train_datagen.flow_from_directory(
    '/Users/qinminxuan/Documents/MLCourse/datasets/validation-horse-or-human/',target_size = (150,150),batch_size = 64,
     class_mode = 'binary')
```

    Found 256 images belonging to 2 classes.



```python
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

hp = HyperParameters()
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Choice("num_filters_layer0", values = [16,64], default = 16),
                                    (3,3),activation = 'relu',input_shape = (150,150,3)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    
    for i in range(hp.Int("num_conv_layers",1,3)):
        model.add(tf.keras.layers.Conv2D(hp.Choice(f"num_filters_layer{i}", values = [16,64], default = 16),(3,3), activation = 'relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int("hidden_units", 128, 512, step = 32),activation = 'relu'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.01), metrics = ['acc'])
    return model
```


```python
tuner = Hyperband(
    build_model, objective = 'val_acc', max_epochs = 10, directory = 'horse-human-params',hyperparameters = hp, project_name
 = 'my-horse-human-project')

tuner.search(train_generator, epochs = 10, validation_data = validation_generator)
```

    
    Search: Running Trial #1
    
    Hyperparameter    |Value             |Best Value So Far 
    num_filters_layer0|64                |?                 
    num_conv_layers   |2                 |?                 
    hidden_units      |160               |?                 
    tuner/epochs      |2                 |?                 
    tuner/initial_e...|0                 |?                 
    tuner/bracket     |2                 |?                 
    tuner/round       |0                 |?                 
    



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-48-3c1a2e16a157> in <module>
          3  = 'my-horse-human-project')
          4 
    ----> 5 tuner.search(train_generator, epochs = 10, validation_data = validation_generator)
    

    ~/opt/anaconda3/envs/tensorflow/lib/python3.8/site-packages/kerastuner/engine/base_tuner.py in search(self, *fit_args, **fit_kwargs)
        129 
        130             self.on_trial_begin(trial)
    --> 131             self.run_trial(trial, *fit_args, **fit_kwargs)
        132             self.on_trial_end(trial)
        133         self.on_search_end()


    ~/opt/anaconda3/envs/tensorflow/lib/python3.8/site-packages/kerastuner/tuners/hyperband.py in run_trial(self, trial, *fit_args, **fit_kwargs)
        352             fit_kwargs['epochs'] = hp.values['tuner/epochs']
        353             fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']
    --> 354         super(Hyperband, self).run_trial(trial, *fit_args, **fit_kwargs)
        355 
        356     def _build_model(self, hp):


    ~/opt/anaconda3/envs/tensorflow/lib/python3.8/site-packages/kerastuner/engine/multi_execution_tuner.py in run_trial(self, trial, *fit_args, **fit_kwargs)
         75     def run_trial(self, trial, *fit_args, **fit_kwargs):
         76         model_checkpoint = keras.callbacks.ModelCheckpoint(
    ---> 77             filepath=self._get_checkpoint_fname(
         78                 trial.trial_id, self._reported_step),
         79             monitor=self.oracle.objective.name,


    ~/opt/anaconda3/envs/tensorflow/lib/python3.8/site-packages/kerastuner/engine/tuner.py in _get_checkpoint_fname(self, trial_id, epoch)
        315             self._get_checkpoint_dir(trial_id, epoch),
        316             'checkpoint')
    --> 317         if (isinstance(self.distribution_strategy, tf.distribute.TPUStrategy) and
        318                 not self.project_dir.startswith('gs://')):
        319             # TPU strategy only support saving h5 format on local path


    AttributeError: module 'tensorflow._api.v2.distribute' has no attribute 'TPUStrategy'


这里注意fstring的用法，这里不仅优化了每一层神经元的个数，还优化了卷积层的层数


```python
best_hps = tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
model.summary()
```

输出结果（来自mooc视频）

![c4p3](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c4p3.jpeg)
