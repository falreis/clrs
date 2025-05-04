import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(tf.constant("Hello, TensorFlow"))
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))
else:
    print("Sorry, no GPU for you...")