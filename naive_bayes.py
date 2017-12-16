#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:54:21 2017

@author: nirorman
"""

import numpy as np
import random
np.set_printoptions(suppress=True)

class naive_bayes:
    def __init__(self):
        path = r'/Users/nirorman/Documents/deepLearning/pima-indians-diabetes.data.txt'
        self.data = self.read_data(path)
        self.train, self.test = self.get_train_test()
        print("train:", self.train)

        self.train_class0 = self.train[np.where(self.train[:,8] == 0)]
        print("train class 0 :", self.train_class0)

        self.train_class1 = self.train[np.where(self.train[:,8] == 1)] 
        print("train class 1 :", self.train_class1)

        self.std_0 = np.std(self.train_class0, axis = 0)
        self.mean_0 = np.mean(self.train_class0, axis = 0)
        self.std_1 = np.std(self.train_class1, axis = 0)
        self.mean_1 = np.mean(self.train_class1, axis = 0)
        
        print("std0 = ", self.std_0)
        print ("mean0 = ", self.mean_0)
        print("std1 = ", self.std_1)
        print ("mean1 = ", self.mean_1)
        
        
    def read_data(self, path):
        return np.genfromtxt(path, delimiter = ',')
        
    def get_train_test(self):
        train_ratio = 0.7
        train_data = []
        test_data = []
        for i in range(len(self.data)):
            if (random.uniform(0, 1) < train_ratio):
                train_data.append(self.data[i])
            else:
                test_data.append(self.data[i])
        return np.array(train_data), np.array(test_data)
    
    def gaussian_vector_density_function(self, X, mu, sigma):
        for i in range(len(X)):
            x_i = X[i]
            f_x_i = self.gaussian_density_function(x_i, mu, sigma)
            
    
    def gaussian_density_function(self, x_i, mu, sigma):
        return (np.e**((-(x_i-mu)**2)/(2*sigma**2)))/np.sqrt(2*np.pi*sigma**2)
    
    def test_density_function(self):
        n = naive_bayes()
        sigma = (np.sqrt(2*np.pi))**(-1)
        mu = 1
        x = 1
        result = n.gaussian_density_function(x, mu, sigma)
        assert(1 == result)
        
        
def main():
    n = naive_bayes()
    n.test_density_function()
    X = (10,139,80,0,0,27.1,1.441,57)
    n.gaussian_vector_density_function(X, n.mean_0, n.std_0)


    

if __name__ == "__main__":
    main()
