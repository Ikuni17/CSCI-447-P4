import numpy as np
import math
import pandas
import random

import matplotlib.pyplot as plt
import experiment


def euclidian_distance(vector_a, vector_b):
    # returns the euclidian distance between the two input vectors
    a = np.array(vector_a)
    b = np.array(vector_b)
    return np.linalg.norm(a - b)


# takes a list of clusters, where a cluster is a list of data points
def calculate_centroids(clusters):
    centroids = []
    for cluster_key in clusters.keys():
        centroids.append(calculate_centroid(clusters[cluster_key]))
    return centroids


def calculate_centroid(cluster):
    cluster = np.array(cluster)
    try:
        cluster_avg = np.zeros_like(cluster[0])
    except(IndexError):
        return np.inf
    for point in cluster:
        # vector addition of all data points to get the sum of each dimension
        cluster_avg = np.add(point, cluster_avg)
    # vector division to get the average value in each dimension
    cluster_avg = np.divide(cluster_avg, len(cluster))

    #print(cluster_avg)
    return cluster_avg


def associate_data(k, centers, data):
    # create the neccessary clusters
    clusters = {}
    for i in range(k):
        clusters[i] = []

    # for each data point, associate it with the cluster that has the nearest centroid
    for point in data:
        min_index = None
        minimum = math.inf

        # find the centroid that is nearest to the current point
        for center_index in range(len(centers)):
            try:
                temp_distance = euclidian_distance(point, centers[center_index])
            except(ValueError):
                continue
            if temp_distance < minimum:
                minimum = temp_distance
                min_index = center_index

        # add the current point to the found nearest cluster
        clusters[min_index].append(point)
    return clusters


def print_vectors(title, input):
    print(title + 'of length ' + str(len(input)))
    for i in input:
        print(i)
    print('\n')


def train(data, k):
    # select k initial centers randomly from data
    centers = select_random_vectors(k, data)

    # create a hash that will not be equal to hash(centers) to enter the while loop
    centers_hash = hash(str([1]))

    # if the last iteration did something, keep going
    while hash(str(centers)) != centers_hash:
        centers_hash = hash(str(centers))

        # associate the data with the nearest centroid and then calculate new centroids
        clusters = associate_data(k, centers, data)
        centers = calculate_centroids(clusters)
    return clusters, centers


# randomly selects n vectors from data
def select_random_vectors(n, data):
    output = []
    for i in range(n):
        output.append(data[int(random.random() * len(data))])
    return output


def read_data(path):
    df = pandas.read_csv(path, header=None)
    return df.values.tolist()


def main():
    data = read_data('datasets/bean.csv')
    k = 10
    clusters, centers = train(data, k)
    for key in clusters.keys():
        x, y = experiment.process_pairs(clusters[key])
        plt.scatter(x, y, linestyle='None', marker=".")
    x, y = experiment.process_pairs(centers)
    plt.scatter(x, y, linestyle='None', marker=".", color='black')
    plt.show()


if __name__ == '__main__':
    main()
