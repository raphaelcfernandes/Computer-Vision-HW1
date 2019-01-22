from scipy.io import loadmat
from scipy import ndimage
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def part1():
    myImages = []
    imagesPath = os.path.abspath(os.path.join("images"))
    filters = os.path.abspath(os.path.join("filters"))
    leung_malik = loadmat(os.path.join(filters, "leung_malik_filter.mat"))["F"]
    images = os.listdir(imagesPath)
    images.sort()
    for i in images:
        img = cv2.imread(os.path.join(imagesPath, i), 0)
        img = cv2.resize(img, (600, 600))
        myImages.append(img)
    if not os.path.exists("part1_plots"):
        os.mkdir("part1_plots")
    for i in range(48):
        fig, axs = plt.subplots(2, 4)
        axs[0, 1].axis("off")
        axs[0, 0].imshow(leung_malik[:,:, i])
        axs[0, 0].set_title("Filter")
        axs[0, 2].imshow(ndimage.convolve(myImages[0],leung_malik[:,:,i]))
        axs[0, 2].set_title(images[0])
        axs[0, 3].imshow(ndimage.convolve(myImages[1],leung_malik[:,:,i]))
        axs[0, 3].set_title(images[1])
        axs[1, 0].imshow(ndimage.convolve(myImages[2],leung_malik[:,:,i]))
        axs[1, 0].set_title(images[2])
        axs[1, 1].imshow(ndimage.convolve(myImages[3],leung_malik[:,:,i]))
        axs[1, 1].set_title(images[3])
        axs[1, 2].imshow(ndimage.convolve(myImages[4],leung_malik[:,:,i]))
        axs[1, 2].set_title(images[4])
        axs[1, 3].imshow(ndimage.convolve(myImages[5],leung_malik[:,:,i]))
        axs[1, 3].set_title(images[5])
        name = 'plot_filter_'+str(i)+'.png'
        fig.savefig(os.path.abspath(os.path.join("part1_plots", name)))
        plt.close(fig)
part1()