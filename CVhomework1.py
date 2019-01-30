from scipy.io import loadmat
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2
import os
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
        img2 = ndimage.convolve(img, F[:, :, i])
        responses[i] = img2
    texture_repr_concat = responses.reshape(
        F.shape[2] * img.shape[0] * img.shape[1])
    texture_repr_mean = []
    for i in range(F.shape[2]):
        texture_repr_mean.append(np.mean(responses[i]))
    return texture_repr_concat, texture_repr_mean


def part3(im1, im2):
    img1 = cv2.imread(im1, 0)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.imread(im2, 0)
    img2 = cv2.resize(img2, (512, 512))
    im1_blur = gaussian_filter(img1, sigma=20)
    im2_blur = gaussian_filter(img2, sigma=1)
    im2_detail = img2 - im2_blur
    hybrid = im1_blur + im2_detail
    cv2.imshow('image', hybrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check(R, y, x):
    if R[y, x] > R[y - 1, x - 1] and R[y, x] > R[y - 1, x] \
            and R[y, x] > R[y - 1, x + 1] and R[y, x] > R[y, x - 1] and R[y, x] > R[y, x + 1] \
            and R[y, x] > R[y + 1, x - 1] and R[y, x] > R[y + 1, x] and R[y, x] > R[y + 1, x + 1]:
        return True

    return False


def extract_keypoints(img):
    k = 0.05
    window_size = 5
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(image,cv2.CV_64F,1,0)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1)
    Ixx = dx**2
    Iyy = dy**2
    R = np.zeros((image.shape[0], image.shape[1]))
    pixel_offset = int(window_size / 2)
    # OpenCV loads height X width. Y x X
    # The foor loop should iterate over Y then over X
    # Det 2x2: (A11 * A22) - (A12 * A21)
    # Trace of a NxN matrix: sum of all elements in the main diagonal
    # R = det(M) - k(trace(M)^2)
    for y in range(2, image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            M = np.zeros((2, 2))
            M[0, 0] = np.sum(Ixx[y - pixel_offset:y + pixel_offset+1, x - pixel_offset:x + pixel_offset+1])
            M[0, 1] = np.sum(dx[y - pixel_offset:y + pixel_offset+1, x - pixel_offset:x + pixel_offset+1]*dy[y - pixel_offset:y + pixel_offset+1, x - pixel_offset:x + pixel_offset+1])
            M[1, 0] = M[0,1]
            M[1, 1] = np.sum(Iyy[y - pixel_offset:y + pixel_offset+1, x - pixel_offset:x +pixel_offset+1])
            R[y, x] = (np.linalg.det(M)) - k*(np.trace(M)**2)
    threshold = 5*abs(np.mean(R))
    #Non-max supression
    # for y in range(1,image.shape[0]-1):
    #     for x in range(1,image.shape[1]-1):
    #         if R[y,x] < threshold:
    #             R[y, x] = 0

    for y in range(1,image.shape[0]-1):
        for x in range(1,image.shape[1]-1):
            if R[y,x] > threshold:
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 255)
                img.itemset((y, x, 2), 0)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    i = loadImages()
    extract_keypoints(cv2.imread(os.path.abspath(
        os.path.join(i.imagesPath, i.myImages[2]))))
    # part3(os.path.abspath(os.path.join(i.imagesPath, i.myImages[0])), os.path.abspath(os.path.join(i.imagesPath, i.myImages[1])))
    # computeTextureReprs(cv2.imread(os.path.abspath(os.path.join(i.imagesPath,i.myImages[0]))),loadmat(os.path.join(i.filters,"leung_malik_filter.mat"))["F"])
