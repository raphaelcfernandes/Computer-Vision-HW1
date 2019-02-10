from scipy.io import loadmat
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from loadImages import loadImages
from operator import itemgetter
import math

def part1():
    loadI = loadImages()
    leung_malik = loadmat(os.path.join(
        loadI.filters, "leung_malik_filter.mat"))["F"]
    if not os.path.exists("part1_plots"):
        os.mkdir("part1_plots")
    myImages = []
    for i in loadI.imagesVector:
        img = cv2.imread(os.path.join(loadI.imagesPath, i), 0)
        img = cv2.resize(img, (100, 100))
        myImages.append(img)
    for i in range(48):
        fig, axs = plt.subplots(2, 4)
        axs[0, 1].axis("off")
        axs[0, 0].imshow(leung_malik[:, :, i])
        axs[0, 0].set_title("Filter")
        axs[0, 2].imshow(ndimage.convolve(myImages[0], leung_malik[:, :, i]))
        axs[0, 2].set_title(loadI.imagesVector[0])
        axs[0, 3].imshow(ndimage.convolve(myImages[1], leung_malik[:, :, i]))
        axs[0, 3].set_title(loadI.imagesVector[1])
        axs[1, 0].imshow(ndimage.convolve(myImages[2], leung_malik[:, :, i]))
        axs[1, 0].set_title(loadI.imagesVector[2])
        axs[1, 1].imshow(ndimage.convolve(myImages[3], leung_malik[:, :, i]))
        axs[1, 1].set_title(loadI.imagesVector[3])
        axs[1, 2].imshow(ndimage.convolve(myImages[4], leung_malik[:, :, i]))
        axs[1, 2].set_title(loadI.imagesVector[4])
        axs[1, 3].imshow(ndimage.convolve(myImages[5], leung_malik[:, :, i]))
        axs[1, 3].set_title(loadI.imagesVector[5])
        name = 'plot_filter_' + str(i) + '.png'
        fig.savefig(os.path.abspath(os.path.join("part1_plots", name)))
        plt.close(fig)


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
    name = 'hybrid' + '.png'
    cv2.imwrite(os.path.abspath(os.path.join(os.path.join(
        "images", "images_for_hybridazation"), name)), hybrid)


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
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    Ixx = dx**2
    Iyy = dy**2
    R = np.zeros((image.shape[0], image.shape[1]))
    pixel_offset = int(window_size / 2)
    location_points = []
    scores = []
    # OpenCV loads height X width. Y x X
    # The foor loop should iterate over Y then over X
    # Det 2x2: (A11 * A22) - (A12 * A21)
    # Trace of a NxN matrix: sum of all elements in the main diagonal
    # R = det(M) - k(trace(M)^2)
    for y in range(pixel_offset, image.shape[0] - pixel_offset):
        for x in range(pixel_offset, image.shape[1] - pixel_offset):
            M = np.zeros((2, 2))
            M[0, 0] = np.sum(Ixx[y - pixel_offset:y + pixel_offset +
                                 1, x - pixel_offset:x + pixel_offset + 1])
            M[0, 1] = np.sum(dx[y - pixel_offset:y + pixel_offset + 1, x - pixel_offset:x + pixel_offset + 1]
                             * dy[y - pixel_offset:y + pixel_offset + 1, x - pixel_offset:x + pixel_offset + 1])
            M[1, 0] = M[0, 1]
            M[1, 1] = np.sum(Iyy[y - pixel_offset:y + pixel_offset +
                                 1, x - pixel_offset:x + pixel_offset + 1])
            R[y, x] = (np.linalg.det(M)) - k * (np.trace(M)**2)
    threshold = 0.01 * abs(R.max())
    # Thresholding pixels
    R[R < threshold] = 0
    # Non-max suppresion
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if check(R, y, x):
                location_points.append((y, x))
                scores.append((R[y, x], (y, x)))
                img.itemset((y, x, 0), 255)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 0)
    sortedScores = sorted(scores, key=itemgetter(0))
    # r = 1
    # for i in sortedScores:
    #     cv2.circle(img, (i[1][1], i[1][0]), r, (0, 0, 255))
    #     r += 1
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return location_points, scores, dx, dy


def detectFeatureKeypoint(x, y, Xmax, Ymax):
    if x - 5 > 0 and x + 5 < Xmax and y - 5 > 0 and y + 5 < Ymax:
        return True
    return False


def compute_features(location_points, scores, Ix, Iy, image):
    features = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = [(x, y) for x, y in location_points if detectFeatureKeypoint(
        x, y, image.shape[0], image.shape[1])]
    featureBin = {'bin1':[],'bin2':[],'bin3':[],'bin4':[],'bin5':[],'bin6':[],'bin7':[]}
    for f in features:
        x, y = f
        Mxy = np.sqrt((Ix[y-5:y+5+1,x-5:x+5+1]**2) + (Iy[y-5:y+5+1,x-5:x+5+1])**2)
        theta = (Ix[y-5:y+5+1,x-5:x+5+1]) / (Iy[y-5:y+5+1,x-5:x+5+1])
        print(theta.shape)
        for i in range(theta.shape[0]):
            for j in theta[i][:]:
                res = math.atan(j)
            break
        break
        


if __name__ == "__main__":
    i = loadImages()
    location_points, scores, Ixx, Iyy = extract_keypoints(cv2.imread(
        os.path.abspath(os.path.join(i.imagesHybridazation, i.imagesHybridazationVector[0]))))
    compute_features(location_points, scores, Ixx, Iyy, cv2.imread(
        os.path.abspath(os.path.join(i.imagesHybridazation, i.imagesHybridazationVector[0]))))
