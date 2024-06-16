import model_lenet5 as ml
from PIL import Image
import numpy as np

def get_test_data(path):
    '''Get test data and turns it into a vector'''

    path = 'D:/OKC/projects/codes/PythonProjects/handwriting_number_recognition/test_data/' + path
    image = Image.open(path)
    g_image = image.convert('L')
    image_arr = np.array(g_image)
    n_image = image_arr/255
    r_image = n_image.tolist()

    return r_image

def main(path):
    '''Predict which number it is'''

    num = 0
    prob = 0
    data = []
    test_data = get_test_data(path)
    data.append(test_data)
    test_model = ml.create_lenet5()
    test_model.load_weights('D:/OKC/projects/codes/PythonProjects/handwriting_number_recognition/model.h5')
    result = test_model.predict(data)
    print('\nOutputs:',end='')
    for i in range(0,10):
        if i == 0:
            print('['+str(i)+']'+'->'+str(result[0][i]))
        else:
            print('        '+'[' + str(i) + ']' + '->' + str(result[0][i]))

    for i in range(0,10):
        if result[0][i] >= prob:
            prob = result[0][i]
            num = i
    print('\nRecognized number:number '+str(num))
    print('Confidence:'+str(result[0][num]*100)+'%')

if __name__ == '__main__':
    while True:
        path = input('\nTest data path:')
        if path == 'q' or path == 'Q':
            break
        else:
            try:
                main(path)
            except FileNotFoundError:
                print('TEST FILE CAN NOT BE FOUND!')