from scipy.io import loadmat
from scipy import ndimage
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from loadImages import loadImages


def part1():
    if not os.path.exists("part1_plots"):
        os.mkdir("part1_plots")
    # for i in range(48):
    #     fig, axs = plt.subplots(2, 4)
    #     axs[0, 1].axis("off")
    #     axs[0, 0].imshow(leung_malik[:,:, i])
    #     axs[0, 0].set_title("Filter")
    #     axs[0, 2].imshow(ndimage.convolve(myImages[0],leung_malik[:,:,i]))
    #     axs[0, 2].set_title(images[0])
    #     axs[0, 3].imshow(ndimage.convolve(myImages[1],leung_malik[:,:,i]))
    #     axs[0, 3].set_title(images[1])
    #     axs[1, 0].imshow(ndimage.convolve(myImages[2],leung_malik[:,:,i]))
    #     axs[1, 0].set_title(images[2])
    #     axs[1, 1].imshow(ndimage.convolve(myImages[3],leung_malik[:,:,i]))
    #     axs[1, 1].set_title(images[3])
    #     axs[1, 2].imshow(ndimage.convolve(myImages[4],leung_malik[:,:,i]))
    #     axs[1, 2].set_title(images[4])
    #     axs[1, 3].imshow(ndimage.convolve(myImages[5],leung_malik[:,:,i]))
    #     axs[1, 3].set_title(images[5])
    #     name = 'plot_filter_'+str(i)+'.png'
    #     fig.savefig(os.path.abspath(os.path.join("part1_plots", name)))
    #     plt.close(fig)

def computeTextureReprs(image, F):
    k = loadImages()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    responses = np.ndarray(shape=(F.shape[2], img.shape[0], img.shape[1]))
    for i in range(F.shape[2]):
        img2 = ndimage.convolve(img, F[:,:, i])
        responses[i] = img2
    texture_repr_concat = responses.reshape(F.shape[2] * img.shape[0] * img.shape[1])
    texture_repr_mean = []
    for i in range(F.shape[2]):
        texture_repr_mean.append(np.mean(responses[i]))
    return texture_repr_concat, texture_repr_mean
    # cv2.imshow('image', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
i = loadImages()
computeTextureReprs(cv2.imread(os.path.abspath(os.path.join(i.imagesPath,i.myImages[0]))),loadmat(os.path.join(i.filters,"leung_malik_filter.mat"))["F"])

# if __name__ == "__main__":
    # part1()
