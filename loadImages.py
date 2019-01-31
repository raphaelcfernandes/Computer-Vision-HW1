import os


class loadImages:
    imagesVector = []
    imagesPath = os.path.abspath(os.path.join("images"))
    filters = os.path.abspath(os.path.join("filters"))
    clusters = os.path.abspath(os.path.join("clusters"))
    imagesHybridazation = os.path.abspath(os.path.join(
        imagesPath, "images_for_hybridazation"))
    imagesHybridazationVector = []

    def __init__(self):
        self.fillArrayOfImages()
        self.fillArrayOfImagesForHybridazation()

    def fillArrayOfImages(self):
        images = os.listdir(self.imagesPath)
        images.sort()
        for i in images:
            if not os.path.isdir(os.path.abspath(os.path.join(self.imagesPath, i))):
                self.imagesVector.append(i)

    def fillArrayOfImagesForHybridazation(self):
        images = os.listdir(self.imagesHybridazation)
        images.sort()
        for i in images:
            if not os.path.isdir(os.path.abspath(os.path.join(self.imagesHybridazation, i))):
                self.imagesHybridazationVector.append(i)
