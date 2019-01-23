import os

class loadImages:
    myImages = []
    imagesPath = os.path.abspath(os.path.join("images"))
    filters = os.path.abspath(os.path.join("filters"))
    
    def __init__(self):
        self.fillArrayOfImages()

    def fillArrayOfImages(self):    
        images = os.listdir(self.imagesPath)
        images.sort()
        for i in images:
            self.myImages.append(i)