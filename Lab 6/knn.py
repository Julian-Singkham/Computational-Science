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
        
        :param self: The KNN object
        :param int k: Number of neighbors
 
        :return: NONE
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        
        :param self: The KNN object
        :param Numpy array X: Datapoint values
        :param Numpy array y: Datapoint classification
 
        :return: NONE
        """
        self.X = X
        self.y = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        
        :param self: The KNN object
        :param Numpy array X: Reference Datapoint values to test
 
        :return: Numpy array of the predictions for each data point
        """
        #0 = setosa
        #1 = versicolor
        #2 = virginica
        predictions = []
        
        #Iterate through each query (row) of the given datatable (2-D array)
        for query in X:
            #Calculate the distance between the query point to every other point
            #This implementation assumes sepal and petal are equally weighted and combines the results
            distance = []
            for row_number in range(len(self.X)):
                #distance = sqrt(a^2 + b^2)
                dist = math.sqrt((query[0] - self.X[row_number][0])**2 + (query[1] - self.X[row_number][1])**2)
                distance.append([dist, self.y[row_number]])
                #This if block is so the code will run with test_blob_classification_loop()
                if(np.shape(self.X)[1] > 2):
                    dist = math.sqrt((query[2] - self.X[row_number][2])**2 + (query[3] - self.X[row_number][3])**2)
                    distance.append([dist, self.y[row_number]])
           
            #Sort the nearest list by the distance (col 0) from low to high
            nearest = sorted(distance, key = lambda col: (col[0],col[1]))
            
            #Find the K closest reference points.
            #This version takes some creative liberties with small values of K because of rounding errors.
            #If the distances of the closests points are exactly the same (either due to rounding errors or just being the same distance)
            #then K should be expanded until the value of nearest[0] is not equal to nearest[k].
            #This ensures that the data is properly classified because limiting values by a small K doesn't give the whole picture
            #due to favoritism towards distance caclulations that occur first.
            #This phenomenon only exists for small values of k (<10).
            nearest_k = [nearest[0][0]]
            for dist in nearest:
                if dist[0] != nearest_k[0] and len(nearest_k) >= self.k:
                    break
                nearest_k.append(dist[1])
                
            #Classification is based on the most common class in the output array
            #List length of one can not be hashed so the first value of the list is taken instead.
            if(len(nearest) == 1):
                target = nearest_k[0]
            else:
                target = max(set(nearest_k), key = nearest_k.count)
            predictions.append(target)
        return np.array(predictions)

    def predict_numpy(self, X):
        """
        Predicts the output variable's values for the query points X using numpy (no loops).
        
        :param self: The KNN object
        :param Numpy array X: Reference Datapoint values to test
 
        :return: Numpy array of the predictions for each data point
        """
        #Calculate the distance between the query point and every other point
        #Distance is a 2D numpy array where each row is the query point and each column is the distance between the other points
        #Shape: (# of query points)x(# of reference points).
        distance = spatial.distance.cdist(X, self.X, 'euclidean')
        
        #Sort the distance array by closest to farthest
        #Each element is the index of the reference point associated with the distance
        #Shape: (# of query points)x(# of reference points).
        nearest = np.argsort(distance)

        #Take the K closest reference points in nearest
        #Since the nearest array contains the index of the reference point, the classification value can be derivied by calling
        #classification[index]
        #Shape: (# of query points)x(k).
        nearest_k = np.array(self.y[nearest[:,:self.k]])
        
        #Return a 1-d array which will contain the predicted flower type for every query point.
        #Stats mode returns a tall single column array so it has to be rotated by 90 degrees to be useable.
        #When stats.mode is called, it returns the most common value and the count. We only care about the value so [0] is added
        #When rot90 is called it returns a 2-D array containing only 1 row so [0] is called to convert to 1-D array
        #Shape: (# of query points)
        return np.rot90(stats.mode(nearest_k, axis=1)[0])[0]