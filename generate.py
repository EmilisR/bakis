import numpy as np
from PIL import Image as img
from os import listdir
from os.path import isfile, join
from random import randint
import os.path
import numpy

letters_path = "d:\\bakalaurinis\\generating\\images\\letters"
numbers_path = "d:\\bakalaurinis\\generating\\images\\numbers"
test_data_path = "D:\\BAKALAURINIS\\ocrd-train\\data\\"

letters = [join(letters_path, f) for f in listdir(letters_path) if isfile(join(letters_path, f))]
numbers = [join(numbers_path, f) for f in listdir(numbers_path) if isfile(join(numbers_path, f))]

imgs    = [ img.open(i) for i in letters + numbers ]

def generate(symbols, name):
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in symbols])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in symbols ) )
    imgs_comb = img.fromarray( imgs_comb)
    imgs_comb.save(test_data_path + name + ".tif")
    with open(test_data_path + name + ".gt.txt", "w") as text_file:
        print(name.upper(), file=text_file)

for x in range(0, 100000):
    letter_1 = imgs[randint(0, len(letters)-1)]
    letter_2 = imgs[randint(0, len(letters)-1)]
    letter_3 = imgs[randint(0, len(letters)-1)]
    number_1 = imgs[randint(21, len(numbers)+len(letters)-1)]
    number_2 = imgs[randint(21, len(numbers)+len(letters)-1)]
    number_3 = imgs[randint(21, len(numbers)+len(letters)-1)]
    symbols = [letter_1, letter_2, letter_3, number_1, number_2, number_3]
    name = os.path.basename(letter_1.filename).split('.')[0] + 
            os.path.basename(letter_2.filename).split('.')[0] + 
            os.path.basename(letter_3.filename).split('.')[0] + 
            os.path.basename(number_1.filename).split('.')[0] + 
            os.path.basename(number_2.filename).split('.')[0] + 
            os.path.basename(number_3.filename).split('.')[0]
    generate(symbols, name)