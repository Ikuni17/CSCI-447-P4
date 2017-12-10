import CL
import KM
import pandas

min_neighbors = 4

def DBScan(data, min_distance):
	labels = [None] * len(data)
	clusters = []
	cluster_id = 0
	for point in range(len(data)):
		cluster = []
		if labels[point] == None:
			expand_cluster(point, data, labels, min_distance, cluster_id, cluster)
			cluster_id += 1
		if cluster:
			clusters.append(cluster)
	return clusters

def expand_cluster(point, data, labels, min_distance, cluster_id, cluster):
	neighbors = get_neighbors(data[point], data, min_distance)
	if len(neighbors) > 4:
		labels[point] = cluster_id
		cluster.append(data[point])
		for neighbor in neighbors:
			if labels[neighbor] == None:
				expand_cluster(neighbor, data, labels, min_distance, cluster_id, cluster)
	elif len(neighbors) > 0:
		for neighbor in neighbors:
			if labels[neighbor] != None and labels[neighbor] == cluster_id:
				labels[point] = cluster_id
				cluster.append(data[point])



def get_neighbors(point, data, min_distance):
	neighbors = []
	for i in range(len(data)):
		if KM.euclidian_distance(point, data[i]) < min_distance:
			neighbors.append(i)
	return neighbors


def get_group(point, data, min_distance):
	group = []
	return group


def read_data(path):
	df = pandas.read_csv(path, header=None)
	return df.values.tolist()


if __name__ == '__main__':
	data = read_data('datasets/airfoil.csv')
	min_distance = 500
	dbscan = DBScan(data, min_distance)
	print(type(dbscan))
	for i in dbscan:
		print('cluster:')
		for point in i:
			print(point)
