import math
import random


def print_vectors(input):
	for i in range(len(input)):
		print(str(input[i]))
	print('\n')


def random_vectors(n, d):
	out = []
	for i in range(n):
		temp = []
		for j in range(d):
			temp.append(random.random())
		out.append(temp)
	return out


def select_random_vectors(n, data):
	output = []
	for i in range(n):
		output.append(data[random.random() * len(data)])
	return output


def update_winner(input, winner, epsilon):
	for i in range(len(winner)):
		winner[i] += (epsilon * (input[i] - winner[i]))

def evaluate_clusters():
	pass


def euclidian_distance(vector_a, vector_b):
	if len(vector_a) == len(vector_b):
		distance = [(a - b)**2 for a, b in zip(vector_a, vector_b)]
		distance = math.sqrt(sum(distance))
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



def train(data, num_clusters, epsilon, epsilon_step):
	dimensions = len(data[0])
	num_data_points = len(data)

	reference_vectors = select_random_vectors(num_clusters, data)
	print('Starting: ')

	while epsilon > 0:
		if index >= len(data):
			index = 0
		print_vectors(reference_vectors)
		compete(data[index], reference_vectors, epsilon)
		index += 1
		epsilon -= epsilon_step

	print('Results')
	print_vectors(reference_vectors)

	for i in range(len(centers)):
		print(str(euclidian_distance(centers[i], reference_vectors[i])))


def main():
	print(str(generate_random_vectors(4, 2, 10)))

if __name__ == '__main__':
	main()
