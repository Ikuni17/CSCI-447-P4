import numpy as np
from copy import deepcopy
import math
import pandas
import random
import pprint


def euclidian_distance(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)
    # print(str(np.linalg.norm(a-b)))

    return np.linalg.norm(a - b)


# takes a list of clusters, where a cluster is a list of data points
def calculate_centroids(clusters):
    centroids = []

    for cluster_key in clusters.keys():
        centroids.append(calculate_centroid(clusters[cluster_key]))

    print(centroids)
    return centroids


def calculate_centroid(cluster):
    cluster = np.array(cluster)
    cluster_avg = np.zeros_like(cluster)
    for point in cluster:
        # print(str(point))
        cluster_avg = np.add(point, cluster_avg)
    cluster_avg = np.divide(cluster_avg, len(cluster))

    return cluster_avg


def associate_data(k, centers, data):
    clusters = {}
    for i in range(k):
        clusters[i] = []

    # print(centers)

    print("Starting Assoc:", len(centers))
    for point in data:
        min_index = None
        minimum = math.inf

        for center_index in range(len(centers)):
            if len(centers[center_index]) == 0:
                continue
            else:
                try:
                    temp_distance = euclidian_distance(point, centers[center_index])
                except(ValueError):
                    print("Point:", point)
                    print("Center:", centers[center_index])
                    exit()
                if temp_distance < minimum:
                    minimum = temp_distance
                    min_index = center_index

        '''if min_index not in clusters:
            clusters[min_index] = [point]
        else:'''
        clusters[min_index].append(point)
    # pprint.pprint(clusters)
    print("End Assoc:", len(clusters.keys()))
    return clusters


def print_vectors(title, input):
    print(title + 'of length ' + str(len(input)))
    for i in input:
        print(i)
    print('\n')


def train(data, k):
    # select k initial centers randomly from data
    centers = select_random_vectors(k, data)
    print("Starting Train:", len(centers))
    # print_vectors('Centers:', centers)

    old_centers_hash = hash(str([1]))
    centers_hash = hash(str(centers))
    while old_centers_hash != centers_hash:
        old_centers_hash = hash(str(centers))

        clusters = associate_data(k, centers, data)
        centers = calculate_centroids(clusters)

        centers_hash = hash(str(centers))
        old_centers = tuple(centers)
        # print_vectors('Centers:', centers)

    print("End Train:", len(centers))
    return clusters


def select_random_vectors(n, data):
    output = []
    for i in range(n):
        output.append(data[int(random.random() * len(data))])

    return output


def read_data(path):
    df = pandas.read_csv(path, header=None)

    return df.values.tolist()


def main():
    data = read_data('datasets/machine.csv')
    # data = [[1, 1, 1], [2, 2, 2], [30, 30, 30], [40, 40, 40]]
    k = 4
    train(data, k)


if __name__ == '__main__':
    main()
