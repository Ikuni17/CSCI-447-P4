import math
import random

reference_vectors = []
input_vectors = []
clusters = []


def ingest_data():
	pass


def update_winner(input, winner, epsilon):
	print(str(winner))
	for i in range(len(winner)):
		winner[i] += (epsilon * (input[i] - winner[i]))
	print(str(winner))

def evaluate_clusters():
	pass


def euclidian_distance(vector_a, vector_b):
	if len(vector_a) == len(vector_b):
		distance = [(a - b)**2 for a, b in zip(vector_a, vector_b)]
		distance = math.sqrt(sum(distance))
		print(distance)
		return distance
	else :
		print('Incompatible vectors')


def compete(input, reference_vectors, epsilon):
	min_index = 0
	min = euclidian_distance(input, reference_vectors[0])

	# find which of the reference vectors is the most similar to (input)
	for i in range(len(reference_vectors)):
		temp_distance = euclidian_distance(input, reference_vectors[i])
		if temp_distance < min:
			min = temp_distance
			min_index = i

	update_winner(input, reference_vectors[min_index], epsilon)


def generate_data(n, len):
	data = []
	for i in range(n):
		temp = []
		for i in range(len):
			temp.append(random.randrange(50))
		data.append(temp)
	return data


def main():
	# tunables
	epsilon = .5
	epsilon_step = 0.01

	reference_vectors = [[0, 0], [0, 0]]

	data = generate_data(10, 2)

	for i in range(len(data)):
		compete(data[i], reference_vectors, epsilon)

if __name__ == '__main__':
	main()
