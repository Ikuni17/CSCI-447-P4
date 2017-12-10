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


def gen_data():
    return np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                      (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                      (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def main():
    csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    datasets = load_datasets(csv_names)

    '''for name in csv_names:
        aco = ACO.ACO(data=datasets[name])
        aco.main(name)'''

    '''clusters = KM.train(gen_data(), 5)
    print(clusters)
    graph2dClusters(clusters)'''
    test_KM(datasets, csv_names)


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

        # Calculate the average in each dimension for the cluster
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
