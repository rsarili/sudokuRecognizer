import numpy as np
import glob
import cv2
import os
import SudokuRecognizer as sr
from mnist_web import mnist

# local packages
import knn

# for time profiling
from timeit import default_timer as timer

def getIntegerLabels(labels):
    """
    Returns data set labels in integer format.
    """
    digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    integerLabels = []
    for i in range(len(labels)):
        integerLabels.append(digits[labels[i]][0])

    return integerLabels


def updateConfusionMatrix(confusionMatrix, results):
    """
    Updates confusion matrix with given results
    """
    for result in results:
        classified, actual, _ = result
        if(classified > -1):
            confusionMatrix[actual][classified] += 1

    return confusionMatrix


def printConfusionMatrixAndPercentages(confusionMatrix):
    """
    Prints confusion matrix and percentages of success of digit predictions
    """
    print('\033[4m'+'class-> %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d' % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) + '\033[0m')
    for i in range(10):
        print('%5d|  ' % i, end='')
        for j in range(10):
            print('%5d ' % confusionMatrix[i][j], end='', flush=True)
        print()

    print('found percentages')
    for k in range(10):
        digitCount = sum(confusionMatrix[k, ])
        truePositive = confusionMatrix[k][k]
        miss = digitCount - truePositive
        print('%d: ' % k, end='')
        if digitCount > 0:
            print('%6.2f%% found, found count: %5d miss count: %5d' % (100 * truePositive / digitCount, truePositive,
                                                                        miss))
        else:
            print(' - ')


sudoku_dataset_dir = '.'
MNIST_dataset_dir = '.'
image_dirs = sudoku_dataset_dir + '/images/*.jpg'
data_dirs = sudoku_dataset_dir + '/images/*.dat'
IMAGE_DIRS = glob.glob(image_dirs)
DATA_DIRS = glob.glob(data_dirs)
IMAGE_DIRS.sort()
DATA_DIRS.sort()

train_images, train_labels, test_images, test_labels = mnist(path=MNIST_dataset_dir)
trainLabels = getIntegerLabels(train_labels)
testLabels = getIntegerLabels(test_labels)


def runMnistTestDataDigitRecognition(threshold, knNeighbor):
    """
    Applies PCA on train data, tries to classify digits in test data using labels. Percentage threshold is given to
    calculate component count. kn neighbor algorithm is used to classify digits. Prints results after operation.
    """
    print('Going to apply PCA and classification on MNIST test data')
    print('In progress...')
    time = timer()
    trainingDataReduced, testDataReduced, dimension = sr.mnistPCA(train_images, test_images, threshold)
    print('...')
    classifications = knn.getClassifiedDigits(trainingDataReduced, trainLabels, testDataReduced, knNeighbor)
    results = [(classifications[i], testLabels[i], classifications[i] == testLabels[i]) for i in range(len(classifications))]

    success = list(filter(lambda x: x[2], results))
    successRate = (float(len(success)) / len(results)) * 100

    print('Finished.')
    print(
        'threshold: %3d%% || dimension: %3d || test count: %5d || max nearest neighbor: %5d\nsuccess rate: %5.2f%% || '
        'time: %.4f s' % (threshold, dimension, len(test_images), knNeighbor, successRate, timer() - time))

    confusionMatrix = np.zeros((10, 10))
    confusionMatrix = updateConfusionMatrix(confusionMatrix, results)
    printConfusionMatrixAndPercentages(confusionMatrix)


def runSudokuDataDigitRecognition(threshold, knNeighbor):
    imageScores = []
    sudokuFound = 0
    cumulativeAcc = 0

    confusionMatrix = np.zeros((10, 10))
    testStartTime = timer()
    print('Calculating eigen pairs ', end='', flush=True)
    time = timer()
    eigenPairs = sr.getEigenPairs(train_images)
    print(' - took %.4f s' % (timer() - time), flush=True)

    time = timer()
    print('Calculating dimension count for threshold %3d' % threshold, end='', flush=True)
    dimensionCount = sr.getDimensionCountForThreshold(threshold, eigenPairs)
    print(' - took %.4f s' % (timer() - time), flush=True)

    time = timer()
    print('Reducing training data to %d dimension' % dimensionCount, end='', flush=True)
    trainImagesReduced = sr.getReducedData(train_images, eigenPairs, dimensionCount)
    print(' - took %.4f s' % (timer() - time), flush=True)

    ## Loop over all images
    for img_dir, data_dir in zip(IMAGE_DIRS, DATA_DIRS):

        # Define your variables etc.:
        image_name = os.path.basename(img_dir)
        data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv2.imread(img_dir)

        try:
            sudokuArray = sr.RecognizeSudoku(img, trainImagesReduced, trainLabels, eigenPairs, dimensionCount,
                                             knNeighbor)
        except KeyboardInterrupt:
             raise
        except:
             print('ERROR image: %s' % image_name)

        # Evaluate Result for current image :
        detectionAccuracyArray = data == sudokuArray
        results = []
        for i in range(9):
            for j in range(9):
                results.append((sudokuArray[i][j], data[i][j], sudokuArray[i][j] == data[i][j]))
        updateConfusionMatrix(confusionMatrix, results)

        accPercentage = np.sum(detectionAccuracyArray) / detectionAccuracyArray.size
        cumulativeAcc = cumulativeAcc + accPercentage

        if sudokuArray[0][0] > -1:
            sudokuFound += 1
            imageScores.append((image_name, float("{0:.2f}".format(accPercentage * 100))))
        print("%13s accuracy : %6.2f%%" % (image_name, accPercentage * 100))


    printConfusionMatrixAndPercentages(confusionMatrix)

    # Average accuracy over all images in the dataset :
    averageAcc = cumulativeAcc / len(IMAGE_DIRS)
    averageAccOnFoundSudoku = cumulativeAcc / sudokuFound

    print(
         'threshold: %3d%% || dimension: %3d || max nearest neighbor: %2d || dataset performance: %5.2f%%\n'
         'sudoku found: %d/%d || dataset performance on found sudoku: %5.2f%% || time: %.4f s'
         % (threshold, dimensionCount, knNeighbor, (100 * averageAcc), sudokuFound,
            len(IMAGE_DIRS), (100 * averageAccOnFoundSudoku), (timer() - testStartTime)))

    imageScores.sort(key=lambda e: e[1], reverse=True)
    bests = imageScores[:5]
    print('best 5: ', bests)
    worsts = imageScores[-5:]
    worsts.sort(key=lambda e: e[1])
    print('worst 5: ', worsts)


threshold = 80
knNeighbor = 5

runMnistTestDataDigitRecognition(threshold, knNeighbor)
runSudokuDataDigitRecognition(threshold, knNeighbor)
