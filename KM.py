import numpy as np
from copy import deepcopy
import pandas
import random

def euclidian_distance(vector_a, vector_b):
	a = np.array(vector_a)
	b = np.array(vector_b)
	# print(str(np.linalg.norm(a-b)))
	return np.linalg.norm(a-b)

# takes a list of clusters, where a cluster is a list of data points
def calculate_centroids(clusters):
	centroids = []
	for cluster in clusters.keys():
		cluster_avg = np.zeros_like(clusters[cluster])
		for point in clusters[cluster]:
			cluster_avg += np.array(point)
		cluster_avg = np.divide(cluster_avg, clusters[cluster])
		centroids.append(cluster_avg)
	return centroids


def associate_data(centers, data):
	clusters = {}
	for point in data:
		min_index = 0
		min = euclidian_distance(point, centers[0])

		for center in range(len(centers)):
			temp_distance = euclidian_distance(point, centers[center])
			if temp_distance < min:
				min = temp_distance
				min_index = center

		if min_index not in clusters:
			clusters[min_index] = [point]
		else:
			clusters[min_index].append(point)
	return clusters


def print_vectors(title, input):
	print(title)
	for i in input:
		print(i)
	print('\n')


def train(data, k):
	# select k initial centers randomly from data
	centers = select_random_vectors(k, data)
	print_vectors('Centers:', centers)

	old_centers = np.zeros_like(centers)
	while not np.array_equal(centers, old_centers):
		old_centers = deepcopy(centers)
		clusters = associate_data(centers, data)
		centers = calculate_centroids(clusters)

		print_vectors('Centers:', centers)


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
	k = 4
	train(data, k)

if __name__ == '__main__':
	main()
