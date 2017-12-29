import cv2
import numpy as np
import math

# local packages
import pcaUtils as pca
import normalizer
import knn

# for time profiling
from timeit import default_timer as timer


def mnistPCA(trainData, testData, threshold):
    """
    Applies PCA on train data, reduces both train data and test data according to given threshold.
    """
    print('Calculating eigen pairs vector', end='', flush=True)
    time = timer()
    trainDataZeroMean = pca.getZeroMean(trainData)
    covarianceMatrix = pca.getCovarianceMatrix(trainDataZeroMean)
    eigenPairs = pca.getEigenValAndVectors(covarianceMatrix)
    eigenVals = [eigenPair[0] for eigenPair in eigenPairs]
    print(' - took %.4f s' % (timer() - time), flush=True)

    print('Calculating cumulative sums', end='', flush=True)
    time = timer()
    cumSums = pca.getCumSum(eigenVals)
    dimension = pca.getComponentCounts([threshold], cumSums)[0]
    print(' - took %.4f s' % (timer() - time), flush=True)

    print('Reducing training data to %d dimension' % dimension, end='', flush=True)
    time = timer()
    trainingDataReduced = pca.transform(trainDataZeroMean, eigenPairs, dimension)
    print(' - took %.4f s' % (timer() - time))
    print('Reducing     test data to %d dimension' % dimension, end='', flush=True)
    time = timer()
    testDataReduced = pca.transform(pca.getZeroMean(testData), eigenPairs, dimension)
    print(' - took %.4f s' % (timer() - time))

    return trainingDataReduced, testDataReduced, dimension


def RecognizeSudoku(img, trainImages, trainLabels, eigenPairs, dimensionCount, knNeighbor):
    """
    Finds Sudoku image, finds inner rectangles, extracts digits in mnist format and classify digits.
    Returns a matrix represents the Sudoku image 0, is used for empty cells.
    If Sudoku is not found matrix is filled with -1.
    """
    onlySudoku, sudokuContour = detectSudoku(img)

    # vertical and horizontal Hough lines are obtained
    verticalLines, horizontalLines = getLines(onlySudoku)

    # lines are near in a specific range are grouped together
    verticalClusters = clusterVerticalLines(verticalLines)
    horizontalClusters = clusterHorizontalLines(horizontalLines)

    # find left right top and bottom of sudoku
    contourLeft, contourRight, contourTop, contourBottom = getContourExtremeValues(sudokuContour)

    # average line calculated for every line group
    averageVerticalLines = findAverageVerticalLines(verticalClusters, contourTop[1], contourBottom[1])
    averageHorizontalLines = findAverageHorizontalLines(horizontalClusters, contourLeft[0], contourRight[0])

    # there should be 10 lines both horizontal and vertical. If there are more try to reduce them.
    # Nothing is done if less lines are found
    if len(averageVerticalLines) > 10:
        removeRedundantVerticalLines(averageVerticalLines)
    if len(averageHorizontalLines) > 10:
        removeRedundantHorizontalLines(averageHorizontalLines)

    # intersection points are found. They should be exact 100
    points = findAllIntersections(averageVerticalLines, averageHorizontalLines)

    # pre process image before try to find digits. Adjust brightness and fix color space.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = adjustBrightness(img)
    # crop inner rectangles of Sudoku using intersection points.
    # If there are no 100 lines, return -1 to show Sudoku is not found.
    croppedImgs = []
    if len(points) == 100:
        # inner rectangles are drawn
        rectanglesPoints = getRectanglesPoints(points)
        for points in rectanglesPoints:
            croppedImg = img[points[0][1]: points[1][1], points[0][0]: points[1][0]]
            croppedImgs.append(croppedImg)
    else:
        print('Points are not found')
        return np.full((9, 9), -1)

    # add extracted digits to a dictionary
    sudokuImages = {}
    for i in range(len(croppedImgs)):
        preprocessedImg = normalizer.preprocess(croppedImgs[i])
        if preprocessedImg is not None:
            sudokuImages[i] = preprocessedImg

    # convert found digits to mnist format
    sudokuImagesMnistFormat= np.zeros((len(sudokuImages), 784))
    i = 0
    for key in sudokuImages:
        sudokuImagesMnistFormat[i] = sudokuImages[key].reshape((1, 784))
        i += 1

    # reduce digits according to eigen vectors and component count, then classify digits
    reducedSudokuImages = getReducedData(sudokuImagesMnistFormat, eigenPairs, dimensionCount)
    classifiedDigits = knn.getClassifiedDigits(trainImages, trainLabels, reducedSudokuImages, knNeighbor)

    # create predicted Sudoku matrix
    sudokuArray = np.zeros((81, 1), dtype=np.int)
    i = 0
    for key in sudokuImages:
        sudokuArray[key] = classifiedDigits[i]

        i += 1
    sudokuArray = sudokuArray.reshape((9, 9)).T
    return sudokuArray


def getEigenPairs(data):
    """
    Returns eigen pairs for given data.
    """
    trainDataZeroMean = pca.getZeroMean(data)
    covarianceMatrix = pca.getCovarianceMatrix(trainDataZeroMean)
    eigenPairs = pca.getEigenValAndVectors(covarianceMatrix)

    return eigenPairs


def getDimensionCountForThreshold(threshold, eigenPairs):
    """
    Returns dimension counts for given threshold percentage using eigen pairs.
    """
    eigenVals = [eigenPair[0] for eigenPair in eigenPairs]

    cumSums = pca.getCumSum(eigenVals)
    dimensionCount = pca.getComponentCounts([threshold], cumSums)[0]

    return dimensionCount


def getReducedData(data, eigenPairs, dimensionCount):
    """
    Reduces data to given dimension using eigen pairs.
    """
    reducedData = pca.transform(pca.getZeroMean(data), eigenPairs, dimensionCount)
    return reducedData


def detectSudoku(originalImg):
    """
    Returns image where only sudoku visible image and contour of this sudoku.
    """
    img = cv2.GaussianBlur(originalImg, (5, 5), 0)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessedImg = adjustBrightness(grayImg)

    sudokuContour = findSudokuContour(preprocessedImg)
    if sudokuContour is None:
        raise Exception('No Contour Found')

    # founded contour area is drawn to black image, then 'bitwise and' is used to obtain only sudoku image
    maskBlack = np.zeros(grayImg.shape, np.uint8)
    cv2.drawContours(maskBlack, [sudokuContour], 0, 255, -1)
    sudokuImgOnly = cv2.bitwise_and(preprocessedImg, maskBlack)

    return sudokuImgOnly, sudokuContour


def findSudokuContour(img):
    """
    Finds biggest contour and assume as Sudoku contour. Applies polyDp to make borders more straight.
    """
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 1)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    bestContour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            if area > maxArea:
                maxArea = area
                bestContour = contour

    approxContour = None
    if bestContour is not None:
        peri = cv2.arcLength(bestContour, True)
        approxContour = cv2.approxPolyDP(bestContour, 0.02 * peri, True)
    return approxContour


def getContourExtremeValues(contour):
    """
    Returns contours extreme values.
    """
    left = tuple(contour[contour[:, :, 0].argmin()][0])
    right = tuple(contour[contour[:, :, 0].argmax()][0])
    top = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    return [left, right, top, bottom]


def adjustBrightness(gray):
    """
    Adjusts brightness by dividing image with closing pixels.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    div = np.float32(gray) / close
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    return res


def getLines(img):
    """
    Finds Hough lines and grouped them as vertical and horizontal.
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=50)
    horizontalLines = []
    verticalLines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            sin = abs((y2 - y1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            if sin < 0.05:
                horizontalLines.append(line)
            elif sin > 0.95:
                verticalLines.append(line)
    return [verticalLines, horizontalLines]


def clusterVerticalLines(verticalLines):
    """
    Clusters vertical lines if they are close to each other less than a threshold.
    """
    verticalLines = sorted(verticalLines, key=lambda line: line[0][0])
    threshold = 15
    x1Avg, y1Avg, x2Avg, y2Avg = verticalLines[0][0]
    n = 1
    clusteredLines = []
    currentCluster = []
    currentCluster.append(verticalLines[0])
    for line in verticalLines[1:]:
        x1, y1, x2, y2 = line[0]
        if x1 < x1Avg + threshold and x1 > x1Avg - threshold and x2 < x2Avg + threshold and x2 > x2Avg - threshold:
            x1Avg += (x1 - x1Avg) / n
            x2Avg += (x2 - x2Avg) / n
            n += 1
            currentCluster.append(line)
        else:
            clusteredLines.append(list(currentCluster))
            currentCluster = []
            x1Avg = x1
            x2Avg = x2
            n = 1
            currentCluster.append(line)
    clusteredLines.append(currentCluster)

    return clusteredLines


def clusterHorizontalLines(horizontalLines):
    """
    Clusters horizontal lines if they are close to each other less than a threshold.
    """
    horizontalLines = sorted(horizontalLines, key=lambda line: line[0][1])
    threshold = 15
    x1Avg, y1Avg, x2Avg, y2Avg = horizontalLines[0][0]
    n = 1
    clusteredLines = []
    currentCluster = []
    currentCluster.append(horizontalLines[0])
    for line in horizontalLines[1:]:
        x1, y1, x2, y2 = line[0]
        if(y1 < y1Avg + threshold and y1 > y1Avg - threshold and y2 < y2Avg + threshold and y2 > y2Avg - threshold):
            y1Avg += (y1 - y1Avg) / n
            y2Avg += (y2 - y2Avg) / n
            n += 1
            currentCluster.append(line)
        else:
            clusteredLines.append(list(currentCluster))
            currentCluster = []
            y1Avg = y1
            y2Avg = y2
            n = 1
            currentCluster.append(line)
    clusteredLines.append(currentCluster)

    return clusteredLines


def findAverageVerticalLines(clusters, yTop, yBottom):
    """
    Finds average vertical line for every given cluster and set given top most and bottom most Y values.
    Extreme values are comes from contours border. This is done to complete short lines.
    """
    verticalLines = []
    for lines in clusters:
        totalX1 = totalX2 = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            isYValuesReverse = True if y1 > y2 else False
            if(isYValuesReverse):
                totalX1 += x2
                totalX2 += x1
            else:
                totalX1 += x1
                totalX2 += x2
        x1Avg = int(totalX1 / len(lines))
        x2Avg = int(totalX2 / len(lines))
        verticalLines.append([x1Avg, yTop, x2Avg, yBottom])
    return verticalLines


def findAverageHorizontalLines(clusters, xLeft, xRight):
    """
    Finds average horizontal line for every given cluster and set given extreme left most and right most X values.
    Extreme values are comes from contours border. This is done to complete short lines.
    """
    horizontalLines = []
    for lines in clusters:
        totalY1 = totalY2 = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            totalY1 += y1
            totalY2 += y2
        y1Avg = int(totalY1 / len(lines))
        y2Avg = int(totalY2 / len(lines))
        horizontalLines.append([xLeft, y1Avg, xRight, y2Avg])
    return horizontalLines


def removeRedundantVerticalLines(lines):
    """
    Reduces vertical lines to one if they are closer each other than a threshold.
    """
    x1, _, x2, _ = lines[0]
    x3, _, x4, _ = lines[-1]

    xFirstLine = max(x1, x2)
    xLastLine = min(x3, x4)

    approxInterval = 0.7 * (xLastLine - xFirstLine) / 9

    currentLine = 0
    nextLine = 1
    size = len(lines)
    processedLines = 1
    while(processedLines < size):
        x1, _, x2, _ = lines[currentLine]
        x3, _, x4, _ = lines[nextLine]
        if min(x3, x4) - max(x1, x2) < approxInterval:
            lines.remove(lines[nextLine])
        else:
            currentLine += 1
            nextLine += 1
        processedLines += 1


def removeRedundantHorizontalLines(lines):
    """
    Reduces horizontal lines to one if they are closer each other than a threshold.
    """
    _, y1, _, y2 = lines[0]
    _, y3, _, y4 = lines[-1]

    yFirstLine = max(y1, y2)
    yLastLine = min(y3, y4)

    approxInterval = 0.7 * (yLastLine - yFirstLine) / 9

    currentLine = 0
    nextLine = 1
    size = len(lines)
    processedLines = 1
    while(processedLines < size):
        _, y1, _, y2 = lines[currentLine]
        _, y3, _, y4 = lines[nextLine]
        if min(y3, y4) - max(y1, y2) < approxInterval:
            lines.remove(lines[nextLine])
        else:
            currentLine += 1
            nextLine += 1
        processedLines += 1


def findIntersection(lineA, lineB):
    """
    Finds intersection of given two lines using determinants.
    """
    x1, y1, x2, y2 = lineA
    x3, y3, x4, y4 = lineB
    try:
        denominator = float((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    except ZeroDivisionError:
        return
    return int(x), int(y)


def findAllIntersections(verticalLines, horizontalLines):
    """
    Finds intersections for given vertical and horizontal lines.
    """
    points = []
    for lineX in verticalLines:
        for lineY in horizontalLines:
            x, y = findIntersection(lineX, lineY)
            points.append([x, y])
    return points


def getRectanglesPoints(points):
    """
    Calculates inner Sudoku rectangles left top and right bottom values.
    """
    rectanglesPoints = []
    bottomPointCounter = 0
    for i in range(0, 89):
        bottomPointCounter += 1
        if bottomPointCounter == 10:
            bottomPointCounter = 0
            continue
        leftTop = points[i]
        rightBottom = points[i+11]
        xThreshold = int((rightBottom[0] - leftTop[0]) / 8)
        yThreshold = int((rightBottom[1] - leftTop[1]) / 8)
        rectLeftTop = (leftTop[0] + xThreshold, leftTop[1] + xThreshold)
        rectRightBottom = (rightBottom[0] - yThreshold, rightBottom[1] - yThreshold)

        rectanglesPoints.append([rectLeftTop, rectRightBottom])

    return rectanglesPoints