import numpy as np


def getDistanceMatrix(trainData, testData):
    """
    Returns distance matrix
    """
    dists = np.zeros((len(testData), len(trainData)), dtype=np.uint16)
    for i in range(len(testData)):
        dists[i, :] = np.sum((trainData - testData[i, :])**2, axis=1)
    return dists


def getClassifiedDigits(trainData, trainLabels, testData, knNeighbor):
    """
    Returns classifications of test data using train data and labels. kn neighbor algorithm is used to decide which label
    will be selected.
    """
    distancesForEachData = getDistanceMatrix(trainData, testData)
    results = []
    for i in range(len(distancesForEachData)):
        neighbors = np.argsort(distancesForEachData[i], axis=0)[:knNeighbor]
        neighborClassifications = np.array(trainLabels)[neighbors]
        classification = np.bincount(neighborClassifications).argmax()
        results.append(classification)
    return results
