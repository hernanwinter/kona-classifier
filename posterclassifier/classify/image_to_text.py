from PIL import Image
import pytesseract
import cv2 as cv2
import numpy as np
import re


def convert(path):
    print(path)
    img = cv2.imread(path)

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imwrite('temp2.jpg', img)

    text = pytesseract.image_to_string(Image.open('temp2.jpg'))
    print(text)

    text = re.sub("\n", " ", text)

    return re.sub("[^a-zA-Z0-9 \\'!]+", "", text)


def test():
    path = "../media/replicas.jpg"

    text = convert(path)
    print(text)

#test()
