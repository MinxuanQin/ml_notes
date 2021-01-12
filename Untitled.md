```python
import numpy as np
%config IPCompleter.greedy=True
```


```python
#加载fashion mnist数据集
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
```


```python
print(train_images.shape)
print(test_images.shape)
```

    (60000, 28, 28)
    (10000, 28, 28)



```python
print(train_labels[:5])
#输出前五个元素
```

    [9 0 0 3 0]



```python
import matplotlib.pyplot as plt
plt.imshow(train_images[1])
```




    <matplotlib.image.AxesImage at 0x7fdc16156e50>




    
![png](output_4_1.png)
    



```python
#构建神经元模型
import tensorflow as tf
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

三层结构，第一层作为输入数据，第三层表示输出，一共十个类别所以是十个神经元。

第二层的神经元个数128是随机指定的

示意图如下：

![picture1 for nn](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p1.jpeg)

![picture2 for nn](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p2.jpeg2)
圆圈代表神经元，Relu的特点为输入为正时才有输出；softmax特点为将输出压缩在0和1之间。

可以用以下代码实现：

``
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation = tf.nn.relu))
model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
``


```python
#查看模型
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________


结果的100480 = （28\*28 + 1）\* 128<br>
1290 = (128 + 1) \* 10<br>
1表示bias，这种网络结构叫做全连接的网络结构。


```python
#训练模型，评估效果
#先对数据进行normalization/scaling
train_images_scaled = train_images/255

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)

test_images_scaled = test_images/255
model.evaluate(test_images,test_labels)
```

    Epoch 1/5
    1875/1875 [==============================] - 2s 918us/step - loss: 3.7440 - accuracy: 0.6807
    Epoch 2/5
    1875/1875 [==============================] - 2s 808us/step - loss: 0.7301 - accuracy: 0.7256
    Epoch 3/5
    1875/1875 [==============================] - 2s 856us/step - loss: 0.6426 - accuracy: 0.7602
    Epoch 4/5
    1875/1875 [==============================] - 2s 898us/step - loss: 0.5823 - accuracy: 0.7957
    Epoch 5/5
    1875/1875 [==============================] - 2s 875us/step - loss: 0.5468 - accuracy: 0.8170
    313/313 [==============================] - 0s 630us/step - loss: 0.6120 - accuracy: 0.7570





    [0.6120078563690186, 0.7570000290870667]



optimizer可以用`tf.optimizers.Adam`,loss同理,`tf.losses.sparse_categorical_crossentropy`

Adam很常用，loss是做分类时常用的，sparse代表整数，后续因为此处的output为one_hot(只有一项为1)而采用

将数据转为0到1之间可以有效增强准确率


```python
#0 T-Shirt/top, 1 Trouser/pants, 2 Pullover shirt, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneakers, 8 Bag, 9 Ankle boot
np.argmax(model.predict([[test_images[0]/255]]))
```

    WARNING:tensorflow:Model was constructed with shape (None, 28, 28) for input Tensor("flatten_input:0", shape=(None, 28, 28), dtype=float32), but it was called on an input with incompatible shape (None, 28).



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-f68a9c057a28> in <module>
          1 #0 T-Shirt/top, 1 Trouser/pants, 2 Pullover shirt, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneakers, 8 Bag, 9 Ankle boot
    ----> 2 np.argmax(model.predict([[test_images[0]/255]]))
    

    ~/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in _method_wrapper(self, *args, **kwargs)
         86       raise ValueError('{} is not supported in multi-worker mode.'.format(
         87           method.__name__))
    ---> 88     return method(self, *args, **kwargs)
         89 
         90   return tf_decorator.make_decorator(


    ~/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in predict(self, x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
       1266           for step in data_handler.steps():
       1267             callbacks.on_predict_batch_begin(step)
    -> 1268             tmp_batch_outputs = predict_function(iterator)
       1269             # Catch OutOfRangeError for Datasets of unknown size.
       1270             # This blocks until the batch has finished executing.


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        578         xla_context.Exit()
        579     else:
    --> 580       result = self._call(*args, **kwds)
        581 
        582     if tracing_count == self._get_tracing_count():


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        616       # In this case we have not created variables on the first call. So we can
        617       # run the first trace but we should fail if variables are created.
    --> 618       results = self._stateful_fn(*args, **kwds)
        619       if self._created_variables:
        620         raise ValueError("Creating variables on a non-first call to a function"


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/function.py in __call__(self, *args, **kwargs)
       2417     """Calls a graph function specialized to the inputs."""
       2418     with self._lock:
    -> 2419       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
       2420     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2421 


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/function.py in _maybe_define_function(self, args, kwargs)
       2772           and self.input_signature is None
       2773           and call_context_key in self._function_cache.missed):
    -> 2774         return self._define_function_with_shape_relaxation(args, kwargs)
       2775 
       2776       self._function_cache.missed.add(call_context_key)


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/function.py in _define_function_with_shape_relaxation(self, args, kwargs)
       2703     self._function_cache.arg_relaxed_shapes[rank_only_cache_key] = (
       2704         relaxed_arg_shapes)
    -> 2705     graph_function = self._create_graph_function(
       2706         args, kwargs, override_flat_arg_shapes=relaxed_arg_shapes)
       2707     self._function_cache.arg_relaxed[rank_only_cache_key] = graph_function


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/function.py in _create_graph_function(self, args, kwargs, override_flat_arg_shapes)
       2655     arg_names = base_arg_names + missing_arg_names
       2656     graph_function = ConcreteFunction(
    -> 2657         func_graph_module.func_graph_from_py_func(
       2658             self._name,
       2659             self._python_function,


    ~/.local/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes)
        979         _, original_func = tf_decorator.unwrap(python_func)
        980 
    --> 981       func_outputs = python_func(*func_args, **func_kwargs)
        982 
        983       # invariant: `func_outputs` contains only Tensors, CompositeTensors,


    ~/.local/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in wrapped_fn(*args, **kwds)
        439         # __wrapped__ allows AutoGraph to swap in a converted function. We give
        440         # the function a weak reference to itself to avoid a reference cycle.
    --> 441         return weak_wrapped_fn().__wrapped__(*args, **kwds)
        442     weak_wrapped_fn = weakref.ref(wrapped_fn)
        443 


    ~/.local/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py in wrapper(*args, **kwargs)
        966           except Exception as e:  # pylint:disable=broad-except
        967             if hasattr(e, "ag_error_metadata"):
    --> 968               raise e.ag_error_metadata.to_exception(e)
        969             else:
        970               raise


    ValueError: in user code:
    
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1147 predict_function  *
            outputs = self.distribute_strategy.run(
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:951 run  **
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2290 call_for_each_replica
            return self._call_for_each_replica(fn, args, kwargs)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2649 _call_for_each_replica
            return fn(*args, **kwargs)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1122 predict_step  **
            return self(x, training=False)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py:927 __call__
            outputs = call_fn(cast_inputs, *args, **kwargs)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/sequential.py:277 call
            return super(Sequential, self).call(inputs, training=training, mask=mask)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/network.py:717 call
            return self._run_internal_graph(
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/network.py:888 _run_internal_graph
            output_tensors = layer(computed_tensors, **kwargs)
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py:885 __call__
            input_spec.assert_input_compatibility(self.input_spec, inputs,
        /Users/qinminxuan/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/input_spec.py:212 assert_input_compatibility
            raise ValueError(
    
        ValueError: Input 0 of layer dense is incompatible with the layer: expected axis -1 of input shape to have value 784 but received input with shape [None, 28]




```python
print(test_images[0].shape)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-17-89908515a317> in <module>
    ----> 1 print([test_images[0]].shape)
    

    AttributeError: 'list' object has no attribute 'shape'

