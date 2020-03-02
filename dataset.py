import pathlib
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata

class Dataset:
    def __init__(self, inputPath, className=None, delimiter=",", maxTuplesNumber=None):
        self.name = inputPath.stem
        self.attributes = None
        self.data = None
        self.normalizedData = None
        self.rowNormalizedData = None
        self.classes = None

        self.data_distanceMatrix = None
        self.normalizedData_distanceMatrix = None
        
        self.nDim = None
        self.nEntries = None

        self.featuresSimilarity = None
        self.featuresCorrelation = None
        

        if "[" in self.name and "]" in self.name:
            className = self.name.split("[")[1].replace("]", "")

        with open(inputPath.resolve(), encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            self.attributes = np.array(next(reader))
            self.data = np.array(list(reader))
            if className is not None:
                classColumn = np.where(self.attributes == className)[0][0]
                self.classes = self.data[:,classColumn]
                self.attributes = np.delete(self.attributes, classColumn)
                self.data = np.delete(self.data, classColumn, 1)
            
            self.data = np.array(self.data, dtype=float)
            
            if maxTuplesNumber is not None and maxTuplesNumber < len(self.data):
                n = min(len(self.data), maxTuplesNumber)
                np.random.seed(57)
                randomIndexes = np.random.choice(len(self.data), n, replace=False)
                self.data = self.data[randomIndexes,]
                if self.classes is not None: self.classes = self.classes[randomIndexes]
            self.nEntries = self.data.shape[0]
            self.nDim = self.data.shape[1]

            #self.normalizedData = MinMaxScaler().fit_transform(self.data)
            self.normalizedData = self._minmax(self.data)

            s = np.sum(self.normalizedData, axis=1)
            s = np.where(s==0, 1, s)
            self.rowNormalizedData = np.array(self.normalizedData / s[:, np.newaxis])

            self.data_distanceMatrix = euclidean_distances(self.data)
            self.normalizedData_distanceMatrix = euclidean_distances(self.normalizedData)


            self.featuresSimilarity = np.full((self.nDim, self.nDim), 1.0)
            self.featuresCorrelation = np.full((self.nDim, self.nDim), 1.0)
            for i in range(self.nDim):
                for j in range(self.nDim):
                    self.featuresSimilarity[i,j] = cosine_similarity(self.data[:,i].reshape(1, -1), self.data[:,j].reshape(1, -1))
                    self.featuresSimilarity[j,i] = self.featuresSimilarity[i,j]

                    self.featuresCorrelation[i,j] = np.abs(np.corrcoef(self.data[:,i], self.data[:,j])[0,1])
                    self.featuresCorrelation[j,i] = self.featuresCorrelation[i,j]

            self.neighbors = np.argsort(self.data_distanceMatrix, 1)
            self.neighborsRank = np.apply_along_axis(rankdata, 1, self.data_distanceMatrix)
            
            self.normalizedNeighbors = np.argsort(self.normalizedData_distanceMatrix, 1)
            self.normalizedNeighborsRank = np.apply_along_axis(rankdata, 1, self.normalizedData_distanceMatrix)

    def _minmax(self, data):
        normalizedData = data.copy()
        mins = np.min(normalizedData, 0)
        maxs = np.max(normalizedData, 0)
        for j in range(normalizedData.shape[1]):
            if maxs[j] - mins[j] == 0:
                normalizedData[:,j] = 0
            else:
                normalizedData[:,j] = (normalizedData[:,j] - mins[j]) / (maxs[j] - mins[j])
        return normalizedData

    def getNormalizedData(self, permutation=None):
        if permutation is None:
            return self.normalizedData
        return self.normalizedData[:,permutation]

    def getFeaturesSimilarity(self, permutation=None):
        if permutation is None:
            return self.featuresSimilarity
        return self.featuresSimilarity[permutation,:][:,permutation]

    def getFeaturesCorrelation(self, permutation=None):
        if permutation is None:
            return self.featuresCorrelation
        return self.featuresCorrelation[permutation,:][:,permutation]

    def getFeatures(self, permutation=None):
        if permutation is None:
            return self.attributes
        return self.attributes[permutation]
            