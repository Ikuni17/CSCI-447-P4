import math
import random
import numpy as np
import pandas


def print_vectors(title, input):
	print(title)
	for i in range(len(input)):
		print(str(input[i]))
	print('\n')


def print_clusters(clusters):
	for cluster in clusters.values():
		print_vectors('Cluster', cluster)


def select_random_vectors(n, data):
	# randomly selects n vectors from data
	output = []
	for i in range(n):
		output.append(data[int(random.random() * len(data))])
	return output


def update_winner(input, winner, epsilon):
	# updates winner according to input and epsilon
	for i in range(len(winner)):
		winner[i] += (epsilon * (input[i] - winner[i]))


def euclidian_distance(vector_a, vector_b):
	# returns the euclidian distance between the two inputs
	a = np.array(vector_a)
	b = np.array(vector_b)
	return np.linalg.norm(a-b)


def compete(input, reference_vectors):
	# returns the reference vector with the lowest euclidian distance from the input point
	min_index = 0
	min = euclidian_distance(input, reference_vectors[0])

	# find which of the reference vectors is the most similar to (input)
	for i in range(len(reference_vectors)):
		temp_distance = euclidian_distance(input, reference_vectors[i])
		if temp_distance < min:
			# if a lower distance reference vector was fourn
			min = temp_distance
			min_index = i

	return min_index


def train(data, num_clusters, epsilon_step = 0.0001, epsilon = 1):
	reference_vectors = select_random_vectors(num_clusters, data)

	index = 0
	while epsilon > 0:
		# if we got to the end of data, start at the beginning again
		if index >= len(data):
			index = 0

		# find the reference vector with the lowest euclidian distance to data[index] and update it accordingly
		winner = compete(data[index], reference_vectors)
		update_winner(data[index], reference_vectors[winner], epsilon)

		# go to the next data point and decrement epsilon
		index += 1
		epsilon -= epsilon_step

	# final pass

	clusters = {}
	for point in data:
		# adds each data point to the cluster associated with the nearest reference vector
		winner_index = compete(point, reference_vectors)

		# if that reference vector doesnt have a cluster, make one. else, append it to the cluster
		if winner_index not in clusters:
			clusters[winner_index] = [point]
		else:
			clusters[winner_index].append(point)
	# return the found clusters
	return clusters



def read_data(path):
	df = pandas.read_csv(path, header=None)
	return df.values.tolist()


def main():
	data = read_data('datasets/airfoil.csv')
	num_clusters = 10
	epsilon = 1
	epsilon_step = 0.000001

	# begins training on the above parameters
	train(data, num_clusters, epsilon, epsilon_step)


if __name__ == '__main__':
	main()
