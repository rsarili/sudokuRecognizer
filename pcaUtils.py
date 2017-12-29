import numpy as np


def getCovarianceMatrix(zeroMeanData):
    """
    Calculates covariance matrix from zero mean data.
    """
    covariance = zeroMeanData.T.dot(zeroMeanData)
    return covariance / (zeroMeanData.shape[0]-1)


def getZeroMean(images):
    """
    Returns zero mean data.
    """
    deviations = images - np.mean(images, axis=0)
    return deviations


def getEigenValAndVectors(covariance):
    """
    Returns eigen pairs (eigen value, eigen vector).
    """
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    eigenPairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigenPairs.sort(key=lambda e: e[0])
    eigenPairs.reverse()
    return eigenPairs


def getFeatureVector(eigenPairs, dimensionCount):
    """
    Returns feature vector that contains horizontally sequenced eigen vectors.
    """
    d = len(eigenPairs)
    featureVector = np.hstack([eigenPairs[i][1].reshape(d, 1) for i in range(0, dimensionCount)])
    return featureVector.T


def getCumSum(eigenVals):
    """
    Returns cumulative sums for given eigen values.
    """
    tot = sum(eigenVals)
    varExp = [(i / tot) * 100 for i in sorted(eigenVals, reverse=True)]
    return np.cumsum(varExp)


def getComponentCounts(thresholds, cumSums):
    """
    Returns component counts according to given thresholds and cumulative sum.
    """
    componentCounts = []
    thresholds = thresholds[:]
    for i in range(len(cumSums)):

        if cumSums[i] > thresholds[0]:
            componentCounts.append(i)
            thresholds.pop(0)
            if len(thresholds) == 0:
                break

    return componentCounts


def transform(zeroMeanData, eigenPairs, componentCount):
    """
    Calculates transformation data using zero mean data, eigen pairs and component count.
    """
    featureVector = getFeatureVector(eigenPairs, componentCount)
    finalData = featureVector.dot(zeroMeanData.T).T
    return finalData









