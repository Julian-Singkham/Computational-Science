import random

import math
import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. 
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.X = X
        self.y = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        """
        predictions = []
        for query in X:
            distance = []
            for reference in range(len(self.X)):
                dist = spatial.distance.cdist(np.array([[query[0], query[1]]]), np.array([[self.X[reference][0], self.X[reference][1]]]))
                distance.append([dist[0], self.y[reference]])
            nearest = []
            for index in range(self.k):
                # Arbitrarily large number
                smallest = 100
                smallest_index = -1
                largest_smallest = -1
                for i in range(len(distance)):
                    if (distance[i][0] < smallest) and (distance[i][0] >= largest_smallest):
                        smallest_index = i
                        smallest = distance[i][0]
                largest_smallest = distance[smallest_index][0]
                nearest.append(distance[smallest_index][1])
            setosa_count = 0
            versicolor_count = 0
            virginica_count = 0
            rose_count = 0
            for target in nearest:
                if target == 0:
                    setosa_count += 1
                elif target == 1:
                    versicolor_count += 1
                elif target == 2:
                    virginica_count += 1
                elif target == 3:
                    rose_count += 1
            if (setosa_count >= virginica_count) and (setosa_count >= versicolor_count) and (setosa_count >= rose_count):
                predictions.append(0)
            if (versicolor_count >= virginica_count) and (versicolor_count >= setosa_count) and (versicolor_count >= rose_count):
                predictions.append(1)
            if (virginica_count >= setosa_count) and (virginica_count >= versicolor_count) and (virginica_count >= rose_count):
                predictions.append(2)
            if (rose_count >= virginica_count) and (rose_count >= setosa_count) and (rose_count >= versicolor_count):
                predictions.append(3)
        return np.array(predictions)
        
    def predict_numpy(self, X):
        """
        Predicts the output variable's values for the query points X using numpy (no loops).
        """
        # The distances variable will have a 2d numpy array which contains an array of distances from each reference point for each query point. It's size will be (# of query points) by (# of reference points).
        distances = spatial.distance.cdist(X, self.X)
        # The sorted_indices variable will have a 2d numpy array which contains an array of indices for each query point. These indeces indicate how far away each query point is from every reference point and are sorted in the order from smallest to largest distance. It's size will be (# of query points) by (# of reference points).
        sorted_indices = np.argsort(distances)
        # The closest_types variable will have a 2d numpy array which contains an array of the flower types of the k closest refrence points to each query point. It's size will be (# of query points) by (k value).
        closest_types = np.array(self.y[sorted_indices[:,:self.k]])
        # The return statement will contain a 1 dimensional array which will contain the predicted flower type for each query point. It's size will be (# of query points).
        return np.rot90(stats.mode(closest_types, axis=1)[0])[0]