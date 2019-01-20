from scipy.io import loadmat
import cv2, os
import numpy as np

def part1():
    myImages = []
    imagesPath = os.path.abspath(os.path.join("images"))
    filters = os.path.abspath(os.path.join("filters"))
    leung_malik = loadmat(os.path.join(filters,"leung_malik_filter.mat"))
    for i in os.listdir(imagesPath):
        img = cv2.imread(os.path.join(imagesPath,i),0)
        img = cv2.resize(img, (100, 100))
        myImages.append(img)
    np.convolve(img,leung_malik["F"][0])
    # for i in myImages:
    #     cv2.imshow('image', i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        

part1()