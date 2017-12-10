import numpy as np

def euclidian_distance(vector_a, vector_b):
	a = np.array(vector_a)
	b = np.array(vector_b)
	# print(str(np.linalg.norm(a-b)))
	return np.linalg.norm(a-b)

# takes a list of clusters, where a cluster is a list of data points
def calculate_centroids(clusters):
	for cluster in clusters:

	pass


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


def train(data, k):
	converged = False
	centers = []

	# select k initial centers randomly from data
	for i in range(k):
		centers.append(data[int(random.random() * len(data)]))



		# associate data points with the neatest cluster


# -----------------------------------------------------
	while not converged:
		converged = True

		# initialize clusters
		for i in range(len(data):
			temp_index = None
			min =
			for j in range(len(hidden_layer) - 1):
				# to not include outputs in distance calculation
				# if min == None or np.linalg.norm(data[i][:len(data[i]) -1] - hidden_layer[j][:len(hidden_layer[j]) - 1]
				temp_distance = np.linalg.norm(np.array(data[i]) - np.array(hidden_layer[j].center))
				if temp_distance < min:
					min = temp_distance
					temp_index = j
			clusters[temp_index].append(i)

		# Calculates new centers
		average = [0] * len(clusters[0])
		for i in range(len(clusters) - 1):
			for item in average:
				item = 0
			if i == 1:
				print(average)
			# For every index in cluster
			for j in range(len(clusters[i]) - 1):
				average = [x + y for x, y in zip(average, data[clusters[i][j]])]

			average = [x/len(clusters[i]) for x in average]
			if average != hidden_layer[i].center:
				converged = False
				hidden_layer[i].center = average


def select_random_vectors(n, data):
	output = []
	for i in range(n):
		output.append(data[int(random.random() * len(data))])
	return output


def main():
	pass

if __name__ == '__main__':
	main()
