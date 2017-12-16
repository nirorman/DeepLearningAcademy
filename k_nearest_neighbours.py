#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:54:48 2017

@author: nirorman
"""
import numpy as np
import random
import math
import operator
import collections

class k_nearest:
    def __init__(self):
        self.data = self.read_data()
        self.train_data, self.test_data = self.get_train_test()
        new_point = [7.6, 3.0, 6.6, 2.1, 'Iris-virginica']
        self.k_nearest_neighbours = self.get_k_nearest(new_point, self.train_data, 3)
        print (self.k_nearest_neighbours )
        self.result = self.vote_majority(self.k_nearest_neighbours)  
        print(self.result)
        
    def read_data(self):
        return np.genfromtxt(r'/Users/nirorman/Documents/deepLearning/iris.data.txt', 
                             delimiter = ',',  dtype = None)
    def get_train_test(self):
        train_ratio = 0.7
        train_data = []
        test_data = []
        for i in range(self.data.size):
            if (random.uniform(0, 1) < train_ratio):
                train_data.append(self.data[i])
            else:
                test_data.append(self.data[i])
        return train_data, test_data
    
    def get_distance(self, point_a, point_b):
        a_num_of_features = len(point_a)
        b_num_of_features = len(point_b)
        assert(a_num_of_features ==  b_num_of_features)
        sum = 0
        for i in range(a_num_of_features-1):
            sum += (point_a[i]-point_b[i])**2
        return math.sqrt(sum)
    
    def get_k_nearest(self, new_point, data, k):
        distances = []
        for row_i in range(len(data)):
            distances.append([data[row_i], self.get_distance(data[row_i], new_point)])
        distances.sort(key = operator.itemgetter(1))
        return [item[0] for item in distances[:k]]
    
    def vote_majority(self, k_points):
        votes = []
        for point_i in k_points:
            votes.append(point_i[-1])
        d = dict.fromkeys(votes, 0)
        for point_i in k_points:
            d[point_i[-1]]+=1
        result = collections.OrderedDict(sorted(d.items(), key=lambda t: t[1]))
        return result.items()[0]
        
        
            
            
        
def main():
    near = k_nearest()
    

if __name__ == "__main__":
    main()
