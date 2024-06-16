import tensorflow as tf

def create_lenet5():
    '''create the network LeNet-5'''

    num_model = tf.keras.models.Sequential()

    # Convolutional layer and pooling layer
    num_model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu',input_shape=(32,32,1)))
    num_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    num_model.add(tf.keras.layers.Conv2D(filters=15, kernel_size=(5, 5), activation='sigmoid'))
    num_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    num_model.add(tf.keras.layers.Flatten())

    # Dense layer
    num_model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    num_model.add(tf.keras.layers.Dense(units=84, activation='sigmoid'))
    num_model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    return num_model

if __name__ == '__main__':
    test_model = create_lenet5()
    test_model.summary()