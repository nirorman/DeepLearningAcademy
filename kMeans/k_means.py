from random import uniform

import math
from matplotlib import pyplot as plt


class KMeans(object):
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = self.get_clusters()
        self.data = data

    def get_clusters(self):
        if type(self.data[0]) is float:
            return [uniform(min(self.data), max(self.data)) for i in range(self.k)]
        elif type(self.data[0]) is (float, float):
            return zip([uniform(min(self.data), max(self.data)) for i in range(self.k)], [uniform(min(self.data), max(self.data)) for i in range(self.k)])

    def _cluster_number_to_list_of_points(self):
        cluster_number_to_list_of_points = {}
        for i in range(self.k):
            cluster_number_to_list_of_points[i] = []
        for d in self.data:
            min_distance = self.distance(d, self.clusters[0])
            best_cluster_index = 0
            for i in range(len(self.clusters)):
                dist = self.distance(d, self.clusters[i])
                if dist < min_distance:
                    min_distance = dist
                    best_cluster_index = i
            cluster_number_to_list_of_points[best_cluster_index].append(d)
        return cluster_number_to_list_of_points

    @staticmethod
    def distance(a, b):
        if type(a) is float and type(b) is float:
            return abs(a - b)
        elif type(a) is (float, float) and type(b) is (float, float):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        else:
            print("ERROR")
            return None

    def _calculate_cluster_location(self, cluster_number_to_list_of_points):
        clusters_locations = []
        for key, value in cluster_number_to_list_of_points.items():
            clusters_locations.append(sum(value)/len(value))
        return clusters_locations

    def get_new_clusters_location(self):
        cluster_number_to_list_of_points = self._cluster_number_to_list_of_points()
        self._visualize(cluster_number_to_list_of_points)
        return self._calculate_cluster_location(cluster_number_to_list_of_points)

    @staticmethod
    def _is_converged(new_cluster_locations, old_cluster_locations):
        return new_cluster_locations == old_cluster_locations

    def cluster(self):
        number_of_iterations = 0
        while True:
            number_of_iterations += 1
            new_cluster_locations = self.get_new_clusters_location()
            if self._is_converged(new_cluster_locations, self.clusters):
                break
            self.clusters = new_cluster_locations
            print(number_of_iterations)
        return self.clusters

    def _visualize(self, cluster_number_to_list_of_points):
        #points = [(x, y) for x in cluster_number_to_list_of_points.keys() for y in cluster_number_to_list_of_points.values()]
        points = []
        for x in cluster_number_to_list_of_points.keys():
            for y in cluster_number_to_list_of_points[x]:
                points.append((x,y))
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        p = plt.plot(x_values, y_values,'.')

        plt.show()
