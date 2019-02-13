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
import time


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
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # if img.shape != (100, 100):
    # img = cv2.resize(img, (100, 100))
    responses = np.ndarray(shape=(F.shape[2], img.shape[0], img.shape[1]))
    for i in range(F.shape[2]):
        img2 = ndimage.convolve(img, F[:, :, i])
        responses[i] = img2
    texture_repr_concat = responses.reshape(
        F.shape[2] * img.shape[0] * img.shape[1])
    texture_repr_mean = []
    for i in range(F.shape[2]):
        texture_repr_mean.append(np.mean(responses[i]))
    return texture_repr_concat, np.asarray(texture_repr_mean)


def part3(im1, im2):
    img1 = cv2.imread(im1, 0)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.imread(im2, 0)
    img2 = cv2.resize(img2, (512, 512))
    im1_blur = gaussian_filter(img1, sigma=30)
    im2_blur = gaussian_filter(img2, sigma=2)
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
    Ixy = dx*dy
    R = np.zeros((image.shape[0], image.shape[1]))
    pixel_offset = int(window_size / 2)
    location_points = []
    scores = []
    for y in range(pixel_offset, image.shape[0] - pixel_offset):
        for x in range(pixel_offset, image.shape[1] - pixel_offset):
            M = np.zeros((2, 2))
            M[0, 0] = np.sum(Ixx[y - pixel_offset:y + pixel_offset +
                                 1, x - pixel_offset:x + pixel_offset + 1])
            M[0, 1] = np.sum(Ixy[y - pixel_offset:y + pixel_offset +
                                 1, x - pixel_offset:x + pixel_offset + 1])
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
            else:
                R[y, x] = 0
    R = cv2.dilate(R, None)
    img[R != 0] = [255, 0, 255]
    sortedScores = sorted(scores, key=itemgetter(0))
    for i in sortedScores:
        cv2.circle(img, (i[1][1], i[1][0]), int(
            i[0]/sortedScores[-1][0]*50), (0, 0, 255))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return location_points, scores, dx, dy


def compute_features(location_points, scores, Ix, Iy):
    offset = 5
    featureBins = np.zeros((len(location_points), 8))
    for i, f in enumerate(location_points):
        x, y = f
        if not (x - offset < 0 or x + offset + 1 > (Ix.shape[0] - 1) or y - offset < 0 or y + offset+1 > (Iy.shape[0] - 1)):
            tMatrix = np.zeros((11, 11))
            Mxy = np.sqrt(np.power(Ix[x - offset:x + offset+1, y - offset:y + offset+1],
                                   2) + np.power(Iy[x - offset:x + offset+1, y - offset: y + offset+1], 2))
            MxyN0 = np.where(Mxy != 0)
            index = (MxyN0[0] + x - offset, MxyN0[1] + y - offset)
            tMatrix[MxyN0] = np.degrees(np.arctan(Iy[index], Ix[index]))
            theta_class = np.digitize(tMatrix, np.arange(-90, 90, 180 / 8)) - 1
            for j in range(8):
                featureBins[i][j] = np.sum(Mxy[theta_class == j])
            if np.max(featureBins[i]) - np.min(featureBins[i]) != 0:
                featureBins[i] = (
                    featureBins[i] - np.min(featureBins[i])) / np.max(featureBins[i]) - np.min(featureBins[i])
                featureBins = np.clip(featureBins, 0, 0.2)
                featureBins[i] = (
                    featureBins[i] - np.min(featureBins[i])) / np.max(featureBins[i]) - np.min(featureBins[i])
    return featureBins


def computeBOWRepr(features, means):
    bow_repr = np.zeros(means.shape[0])
    for row, cols in enumerate(features):
        maxV = np.inf
        for index, i in enumerate(means):
            v = np.linalg.norm(cols - i, 2, 0)
            if v < maxV:
                maxV = v
                cluster = index
        bow_repr[cluster] += 1
    bow_repr = bow_repr / np.sum(bow_repr)
    return bow_repr


def part7(loadI):
    imageRepresentations = {}
    cont = 0
    bow_repr_vector = []
    texture_repr_concat_vector = []
    texture_repr_mean_vector = []
    for i in loadI.imagesVector:
        image = cv2.resize(cv2.imread(os.path.abspath(
            os.path.join(loadI.imagesPath, i))), (100, 100))
        location_points, scores, Ix, Iy = extract_keypoints(image)
        features = compute_features(location_points, scores, Ix, Iy)
        bow_repr = computeBOWRepr(features, loadmat(os.path.abspath(
            os.path.join(loadI.clusters, 'means.mat')))['means'])
        texture_repr_concat, texture_repr_mean = computeTextureReprs(image, loadmat(
            os.path.abspath(os.path.join(loadI.filters, 'leung_malik_filter.mat')))['F'])
        bow_repr_vector.append(bow_repr)
        texture_repr_concat_vector.append(texture_repr_concat)
        texture_repr_mean_vector.append(texture_repr_mean)

    # Within class calculation
    bow_repr_within = []
    texture_repr_concat_within = []
    texture_repr_mean_within = []
    for i in range(0, 6, 2):
        bow_repr_within.append(np.linalg.norm(
            bow_repr_vector[i] - bow_repr_vector[i + 1], 2, 0))
        texture_repr_concat_within.append(np.linalg.norm(
            texture_repr_concat_vector[i] - texture_repr_concat_vector[i + 1], 2, 0))
        texture_repr_mean_within.append(np.linalg.norm(
            texture_repr_mean_vector[i] - texture_repr_mean_vector[i + 1], 2, 0))
    # between class calculation
    bow_repr_between = []
    texture_repr_concat_between = []
    texture_repr_mean_between = []
    for class1 in range(0, 6):
        for class2 in range(class1 + 2 if class1 % 2 == 0 else class1 + 1, 6):
            bow_repr_between.append(np.linalg.norm(
                bow_repr_vector[class1] - bow_repr_vector[class2], 2, 0))
            texture_repr_concat_between.append(np.linalg.norm(
                texture_repr_concat_vector[class1] - texture_repr_concat_vector[class2], 2, 0))
            texture_repr_mean_between.append(np.linalg.norm(
                texture_repr_mean_vector[class1] - texture_repr_mean_vector[class2], 2, 0))

    print("Bag of words ratio: ", np.average(
        bow_repr_within) / np.average(bow_repr_between))
    print("Texture concatenated ratio: ", np.average(
        texture_repr_concat_within) / np.average(texture_repr_concat_between))
    print("Texture mean ratio: ", np.average(
        texture_repr_mean_within) / np.average(texture_repr_mean_between))


if __name__ == "__main__":
    i = loadImages()

    # computeTextureReprs(cv2.imread(os.path.abspath(os.path.join(i.imagesPath, i.imagesVector[0]))), loadmat(
    # os.path.abspath(os.path.join(i.filters, 'leung_malik_filter.mat')))['F'])
    # location_points, scores, Ixx, Iyy = extract_keypoints(cv2.imread(
    #     os.path.abspath(os.path.join(i.imagesPath, i.imagesVector[5]))))
    # part3(os.path.abspath(os.path.join(i.imagesHybridazation, i.imagesHybridazationVector[0])), os.path.abspath(
    #     os.path.join(i.imagesHybridazation, i.imagesHybridazationVector[1])))
    # part7(i)
