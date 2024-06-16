import model_lenet5 as ml
from PIL import Image
import numpy as np

def get_train_data(path):
    '''Get the train data and turns it into a vector'''

    image = Image.open(path)
    g_image = image.convert('L')
    image_arr = np.array(g_image)
    n_image = image_arr/255
    r_image = n_image.tolist()

    return r_image

def train():
    '''train the network'''

    x_train = []
    y_train = []
    for i in range(0,10):
        for j in range(0,10):
            path_first = '.\\train_data\\'
            path = path_first + str(i) + str(j) +'.png'
            x_train.append(get_train_data(path))
            y_part = []
            for k in range(0,10):
                if k==i:
                    y_part.append(1)
                else:
                    y_part.append(0)
            y_train.append(y_part)
    train_model = ml.create_lenet5()
    train_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_model.fit(x=x_train, y=y_train, epochs=300, batch_size=50)
    train_model.save('model.h5')

if __name__  ==  '__main__':
    train()
