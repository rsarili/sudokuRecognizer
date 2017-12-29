import cv2
import numpy as np


def preprocess(image):
    """
    Returns a 28x28 image in MNIST format (scaled to 20x20 then centered in 28x28 image) after doing following steps
    - Convert image to binary, close spaces
    - Select nearly middle of the image, calculate average of pixels. Assume there is digit if average is greater than
    a threshold
    - Starting from the center of the image go up, down, left and right directions. Calculate sum of the rows or
    columns, if they are greater keep doing it until reach to end of the image. End values will be borders of the digit
    rectangle
    - Exract digit rectangle according to found rows and columns
    - Scale rectangle to 20x20
    - Center rectangle in 28x28
    """

    # convert image to binary inverse and close spaces
    ret, threshedImaged = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)
    threshedImaged = cv2.morphologyEx(threshedImaged, cv2.MORPH_CLOSE, (3, 3))

    rows = threshedImaged.shape[0]
    cols = threshedImaged.shape[1]

    # select nearly middle of the image. Calculate average of the pixels. Pixels greater than threshold, there is a
    # digit.
    middleImage = threshedImaged[int(rows/6):int(5*rows/6), int(cols/6):int(5*cols/6)]
    avgMiddleImage = np.average(middleImage)
    if avgMiddleImage < 30:
        return None

    rowTop = -1
    rowBottom = -1
    colLeft = -1
    colRight = -1

    rowCenter = int(threshedImaged.shape[0] / 2)
    colCenter = int(threshedImaged.shape[1] / 2)
    # take median values as threshold
    rowThreshold = sorted([sum(row) for row in threshedImaged])[rowCenter] / 2.5
    colThreshold = sorted([sum(col) for col in threshedImaged.T])[colCenter] / 2

    # Starting from the center of the image go up, down, left and right directions. Calculate sum of the rows and
    # columns, if it is greater than a threshold, keep going until the end of the image. End values are
    # borders of the digit rectangle
    for row in threshedImaged:
        sum(row)

    for i in range(rowCenter, threshedImaged.shape[0]):
            rowSum = sum(threshedImaged[i])
            if rowSum <= rowThreshold:
                rowBottom = i
                break

    for i in range(rowCenter, 0, -1):
            rowSum = sum(threshedImaged[i])
            if rowSum <= rowThreshold:
                rowTop = i
                break

    for i in range(colCenter, 0, -1):
            colSum = sum([x[i] for x in threshedImaged])
            if colSum <= colThreshold:
                colLeft = i
                break

    for i in range(colCenter, threshedImaged.shape[1]):
            colSum = sum([x[i] for x in threshedImaged])
            if colSum <= colThreshold:
                colRight = i
                break

    if colLeft == -1 or colRight == -1 or rowTop == -1 or rowBottom == -1 or colLeft == colRight or rowBottom == rowTop:
         return None

    # Extract found digit rectangle
    croppedImage = threshedImaged[rowTop:rowBottom, colLeft:colRight]

    # found scaled height and weight to resize to 20x20
    width = croppedImage.shape[1]
    height = croppedImage.shape[0]
    target = 20
    if width > height:
        scale = target / float(width)
        scaledHeight = int(scale * height)
        scaledWidth = target
    else:
        scale = target / float(height)
        scaledWidth = int(scale * width)
        scaledHeight = target

    # resize to 20x20
    scaledImage = cv2.resize(croppedImage, (scaledWidth, scaledHeight))

    # center digit rectangle in a 28x28 image
    newSize = 28
    newImage = np.zeros((newSize, newSize))
    newX = int((newSize-scaledWidth)/2)
    newY = int((newSize-scaledHeight)/2)

    newImage[newY:newY+scaledHeight, newX:newX+scaledWidth] = scaledImage

    return newImage/255
