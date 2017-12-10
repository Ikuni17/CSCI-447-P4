'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 9, 2017
'''

import ACO
import CL
import DB
import KM
import PSO
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random


# Read in a csv dataset, convert all values to numbers, and return as a 2D list
def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


def load_datasets(csv_names):
    # Dictionary with dataset name as the key and a 2D list of vectors as the value
    datasets = {}

    # Populate the dictionary using helper function
    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\{0}.csv'.format(csv_names[i]))

    return datasets


def gen_data(mu, sigma, cluster_size, magnitude, dimension = 2):
    if len(mu) != len(sigma):
        print('invalid data generation parameters')
    else:
        data = []
        for i in range(len(mu)):
            # print('cluster')
            cluster = np.ndarray.tolist(sigma[i] * np.random.randn(cluster_size, dimension) + mu[i] * magnitude)
            # print(str(cluster))
            for point in cluster:
                data.append(point)
        random.shuffle(data)
        # KM.print_vectors('Data:', data)
        df = pandas.DataFrame(data)
        df.to_csv('data.csv', index=False)
        return data


def process_pairs(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    return x, y


def test_CL():
    cluster_size = 1000
    num_clusters = 4
    epsilon_step = 1
    magnitude = 10

    # mu = []
    # sigma = []
    # for i in range(num_clusters):
    #     mu.append(random.random())
    #     sigma.append(random.random())
    # data = gen_data(mu, sigma, cluster_size, magnitude)

    data = get_dataset('data.csv')


    clusters = CL.train(data, num_clusters, epsilon_step)
    for key in clusters.keys():
        x, y = process_pairs(clusters[key])
        plt.scatter(x, y)

    print(str(cluster_size) + '\n' + str(num_clusters) + '\n' + str(epsilon_step) + '\n' + str(magnitude))
    plt.show()


def main():
    test_CL()
    # print(str(clusters))


    # plt.scatter(x, y)
    # plt.scatter(y, x)
    # plt.show()



    # csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    # datasets = load_datasets(csv_names)

    '''for name in csv_names:
        aco = ACO.ACO(data=datasets[name])
        aco.main(name)'''

    '''clusters = KM.train(gen_data(), 5)
    print(clusters)
    graph2dClusters(clusters)'''
    # test_KM(datasets, csv_names)


def test_KM(datasets, csv_names):
    results = {}
    for name in csv_names:
        results[name] = {2: 0, 3: 0, 4: 0, 5: 0}
        for k in range(2, 6):
            print("Starting test with k={0} on {1}".format(k, name))
            for i in range(25):
                clusters = KM.train(datasets[name], k)
                for key in clusters.keys():
                    if len(clusters[key]) == 0:
                        results[name][k] += 1

    print(results)


def graph2dClusters(data):
    for cluster in data.values():
        xVal = [x[0] for x in cluster]
        yVal = [y[1] for y in cluster]
        plt.scatter(xVal, yVal, linestyle='None', marker=".")

    plt.show()

def evaluate_clusters(clusters):
    centers = []
    average_dist = 0
    num_points = 0
    for cluster in clusters:
        center_point = [0]*len(cluster[0])
        num_points = num_points + len(cluster)

        # Calculate the average in each dimension
        for i in range(len(center_point)):
            dimCut = [dim[i] for dim in cluster]
            center_point[i] = sum(dimCut)/len(dimCut)
        centers.append(center_point)

        # Calcualte the distance from each point to its center
        for point in cluster:
            average_dist = average_dist + KM.euclidian_distance(point, center_point)

    return average_dist/num_points

if __name__ == '__main__':
    main()
