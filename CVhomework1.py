from scipy.io import loadmat
from scipy import ndimage
from scipy.ndimage import gaussian_filter
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

def part3(im1, im2):
    img1 = cv2.imread(im1,0)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.imread(im2,0)
    img2 = cv2.resize(img2, (512, 512))
    im1_blur = gaussian_filter(img1, sigma=20)
    im2_blur = gaussian_filter(img2, sigma=1)
    im2_detail = img2 - im2_blur
    hybrid = im1_blur + im2_detail
    cv2.imshow('image', hybrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_keypoints(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_offset = 2
    k = 0.05
    window_size = 5
    dx, dy = np.gradient(image)
    R = np.zeros((image.shape[0], image.shape[1]))
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy
    #OpenCV loads height X width. Y x X
    #The foor loop should iterate over Y then over X
    #Det 2x2: (A11 * A22) - (A12 * A21)
    #Trace of a NxN matrix: sum of all elements in the main diagonal
    for y in range(pixel_offset, image.shape[0] - pixel_offset):
        for x in range(pixel_offset, image.shape[1] - pixel_offset):
            windowIxx = Ixx[y-pixel_offset:y+pixel_offset+1, x-pixel_offset:x+pixel_offset+1]
    print(windowIxx)
    print(Ixx)



if __name__ == "__main__":
    i = loadImages()
    extract_keypoints(cv2.imread(os.path.abspath(os.path.join(i.imagesPath,i.myImages[0]))))
    # part3(os.path.abspath(os.path.join(i.imagesPath, i.myImages[0])), os.path.abspath(os.path.join(i.imagesPath, i.myImages[1])))
    # computeTextureReprs(cv2.imread(os.path.abspath(os.path.join(i.imagesPath,i.myImages[0]))),loadmat(os.path.join(i.filters,"leung_malik_filter.mat"))["F"])

