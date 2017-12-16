import numpy as np
from sklearn import datasets, model_selection
import scipy

# we'll have 3 different iris types, so we'll have 3 gaussians.
NUMBER_OF_CLASSES = 3


# 6. Write a P function that calculates for each point the probability of belonging to each gaussian, based on the
# mean and variance of this gaussian
def probability(point, gaussian_mean, gaussian_variance):
    # The probability that the given point is in the given gaussian
    return scipy.stats.multivariate_normal.pdf(point, gaussian_mean, gaussian_variance)


def get_N(X_train, covariance_matrix, means_per_gaussian):
    N = np.zeros((len(X_train), NUMBER_OF_CLASSES))
    for i in range(len(X_train)):
        for j in range(NUMBER_OF_CLASSES):
            N[i, j] = probability(X_train[i], means_per_gaussian[j], covariance_matrix)

    return N


def e_step(X_train, covariance_matrix, means_per_gaussian, probability_of_gaussian, responsibility_matrix):
    N_per_point = get_N(X_train, covariance_matrix, means_per_gaussian)
    numerator_matrix_per_point = probability_of_gaussian * N_per_point
    for i in range(len(X_train)):
        denominator = 0
        for g in range(NUMBER_OF_CLASSES):
            denominator += numerator_matrix_per_point[i, g]
        for g in range(NUMBER_OF_CLASSES):
            responsibility_matrix[i, g] = numerator_matrix_per_point[i, g] / denominator


def main():
    # 1. load the dataset and split into test and train:
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

    # 2. randomly choose the starting centroids / means as three of the points from datasets:
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    means_per_gaussian = [np.mean(X_train, axis=0) - np.std(X_train, axis=0), np.mean(X_train, axis=0), np.mean(X_train, axis=0) + np.std(X_train, axis=0)]

    # 3. initialize the variances for each gaussians using the median:
    covariance_matrix = np.cov(X_train, rowvar=False)

    # 4. initialize the probabilities / weights for each gaussians, as equally distributed
    probability_of_gaussian = [1.0/NUMBER_OF_CLASSES, 1.0/NUMBER_OF_CLASSES, 1.0/NUMBER_OF_CLASSES]  # each gaussian has the same amount of data, so they have the same
    # probability to occur

    # 5. Responsibility(membership) matrix is initialized to all zeros, we have responsibility for each of n points for
    # each of k gaussians
    responsibility_matrix = np.zeros((len(X_train), NUMBER_OF_CLASSES))

    # 8. Write the E - step(expectation) in which we multiply this P function for every point by the weight of the corresponsing cluster pass
    iter_number = 100
    for k in range(iter_number):
        e_step(X_train, covariance_matrix, means_per_gaussian, probability_of_gaussian, responsibility_matrix)
        m_step()





if __name__ == '__main__':
    main()
