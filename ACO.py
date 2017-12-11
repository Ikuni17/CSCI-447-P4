'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 9, 2017
'''

from collections import OrderedDict
import math
from matplotlib import pyplot as plt
import numpy as np
import random


# Class which represents an ant, most logic is handled within the grid. The ant only keeps track of its location and
# what its carrying if anything. Probabilities are calculated in the grid and given to an ant for it to determine if
# it should pickup or drop a datum. Location is updated randomly by the grid and the ant is just told where to move
class Ant():
    def __init__(self, location):
        # Current location in the 2D grid where this ant is
        self.location = location
        # The datum the ant is carrying, if any
        self.carrying = None

    # Probabilistically picks up a datum off the grid. Probability is determined by the density of the neighborhood
    def pickup(self, prob, grid):
        if random.random() <= prob:
            # Pickup the datum and remove it from the grid
            self.carrying = grid[self.location]['Datum']
            grid[self.location]['Datum'] = None

    # Probabilistically drops a datum on the grid. Probability is determined by the density of the neighborhood
    def drop(self, prob, grid):
        if random.random() <= prob:
            # Drop the datum and allow this ant to carry another
            grid[self.location]['Datum'] = self.carrying
            self.carrying = None


# Class which represents the main ACO algorithm. Keeps track of the grid and the ants
class ACO():
    def __init__(self, data, patch_size=3, step_size=1, gammas=[50000, 0.1, 0.7]):
        # Find how large the dataset is and determine the grid size and amount of ants based on that
        length = len(data)
        # Make a grid which has 10 times as many locations as data points
        self.grid_size = math.ceil(math.sqrt(length * 10))
        # OrderedDict where the keys are tuples for coordinates (x, y), and the value is a dictionary
        self.grid = self.init_grid(self.grid_size, data)
        # Create twice as many ants as data points
        self.ants = self.init_ants(int(length / 10))
        # The search space of the ants
        # self.patch_size = math.ceil(length * 0.024) 5
        self.patch_size = patch_size
        # The amount of spaces an ant can move per move
        self.step_size = step_size
        # Tunable parameters for probabilities. 0th index is for density function, 1st is threshold for the probability
        # to pickup a datum, 2nd is a threshold for the probability to drop a datum
        self.gammas = gammas

    # Initialize the 2D, then place the data in the grid randomly
    def init_grid(self, dim, data):
        # Use an ordered dictionary so the points are in the order they are created
        grid = OrderedDict()

        # Create all the points in the 2D grid and assign its value with a dictionary
        for x in range(dim + 1):
            for y in range(dim + 1):
                grid[(x, y)] = {'Ant': None, 'Datum': None}

        # Place all the data points randomly in the grid
        for datum in data:
            # Choose a random location
            rand_pos = (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            # Loop until we have a point in the grid which isn't taken
            while grid[rand_pos]['Datum'] is not None:
                rand_pos = (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            # Add the datum to the grid
            grid[rand_pos]['Datum'] = datum

        return grid

    # Create all the ants and place them randomly on the grid
    def init_ants(self, num_ants):
        ants = []
        for i in range(num_ants):
            # Choose a random location
            rand_pos = (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            # Loop until we have a point in the grid which isn't taken
            while self.grid[rand_pos]['Ant'] is not None:
                rand_pos = (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            # Create an ant and append it to the list of ants
            ants.append(Ant(location=rand_pos))
            # Add the ant's reference to the dictionary
            self.grid[rand_pos]['Ant'] = ants[i]

        return ants

    # Find the similarity between two datums based on Euclidean distance
    def similarity(self, x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    # Finds the density of the neighborhood around an ant. Pickup is a boolean if the ant is trying to pickup or drop
    # a datum, to know which datum to compare to the neighborhood
    def density(self, ant, pickup):
        # Find the min and max coordinates so we stay on the grid
        min_x = max(0, ant.location[0] - self.patch_size)
        max_x = min(self.grid_size, ant.location[0] + self.patch_size)
        min_y = max(0, ant.location[1] - self.patch_size)
        max_y = min(self.grid_size, ant.location[1] + self.patch_size)
        # Sum all the similarities
        density_sum = 0

        # Search in the space around the ant
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # If a point has a datum compare it
                if self.grid[(x, y)]['Datum'] is not None:
                    if pickup:
                        # If the ant is trying to pickup, compare to the datum which is on the same point as the ant
                        density_sum += (1 - (self.similarity(self.grid[ant.location]['Datum'],
                                                             self.grid[(x, y)]['Datum']) / self.gammas[0]))
                    else:
                        # If the ant is trying to drop a datum, compare to the datum which it is carrying
                        density_sum += (1 - (
                                self.similarity(ant.carrying, self.grid[(x, y)]['Datum']) / self.gammas[0]))

        return max(0, ((1 / self.patch_size ** 2) * density_sum))

    # Calculate the probability to pickup, based on the density of the neighborhood and tunable params
    def prob_pick(self, ant):
        return (self.gammas[1] / (self.gammas[1] + self.density(ant, True))) ** 2

    # Calculate the probability to drop, based on the density of the neighborhood and tunable params
    def prob_drop(self, ant):
        # Get the density of the neighborhood
        density = self.density(ant, False)
        if density < self.gammas[2]:
            return 2 * density
        # If there is a high density drop the datum
        else:
            return 1

    # Find a valid position for an ant to move to within step size
    def find_valid_pos(self, ant):
        # Only try positions which
        min_x = max(0, ant.location[0] - self.step_size)
        max_x = min(self.grid_size, ant.location[0] + self.step_size)
        min_y = max(0, ant.location[1] - self.step_size)
        max_y = min(self.grid_size, ant.location[1] + self.step_size)
        # Fail safe if an ant is stuck, i.e. its surrounded by ants and cannot move
        i = 0
        max_tries = 100

        # Move the ant to an available spot within step size
        rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
        # Loop until we have a point in the grid which isn't taken
        while self.grid[rand_pos]['Ant'] is not None and i <= (self.step_size * max_tries):
            rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            i += 1

        # If we reached the fail safe, just leave the ant and hope another moves before the next iteration
        if i == (self.step_size * max_tries):
            return ant.location
        else:
            return rand_pos

    # Graphs the points in the 2D grid which currently have a datum
    def graph_grid(self, name, clusters):
        # Use a different color for each cluster
        colors = np.random.rand(len(clusters.keys()))

        # Make a large graph
        plt.figure(figsize=(25.5, 13.5), dpi=100)
        # Iterate through each cluster and add it to the scatterplot
        for cluster in clusters.values():
            x = [x[0] for x in cluster]
            y = [y[1] for y in cluster]
            plt.scatter(x, y)

        # Create a legend of cluster number and color
        plt.legend(clusters.keys())
        plt.grid()
        plt.savefig('tuning\\ACO\\ACO-{2}, {0}-1M, 10%, {1}, C.png'.format(str(self.gammas), self.patch_size, name))
        # plt.show()

    # Search for clusters in the grid, based on depth first search and an area around each point that has datum
    def find_clusters(self):
        # Get the points (x, y) from the grid which are considered a cluster
        clusters_points = self.connect_points()
        clusters_real = {}
        # Make clusters which contain the real values for comparison to other algorithms
        for key in clusters_points.keys():
            clusters_real[key] = []
            for point in clusters_points[key]:
                # Get the datum vector from the grid which corresponds to this point
                clusters_real[key].append(self.grid[(point[0], point[1])]['Datum'])

        # Return both views of the cluster
        return clusters_points, clusters_real

    # Use DFS to find all points which are connected and therefore clustered
    def connect_points(self):
        # Mark each point as unvisited
        visited = [[False for j in range(self.grid_size + 1)] for i in range(self.grid_size + 1)]
        # Keep track of the amount of clusters, used as a dictionary key and identifier
        count = 0
        # Has the structure {id: [(x1, y1), (x2, y2), ...]
        clusters = {}
        # Check every point in the grid
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1):
                # If the point has not been visited and there is a datum, this is a new cluster
                if visited[i][j] is False and self.grid[(i, j)]['Datum'] is not None:
                    # Add a new cluster to the dictionary
                    clusters[count] = []
                    # Start the recursion by calling helper
                    self.DFS(i, j, visited, clusters[count])
                    # Move to the next cluster
                    count += 1

        # List of clusters to remove from the dictionary because they are empty lists
        remove = []

        # Find empty lists in the dictionary, but wait to remove or runtime errors will occur
        for k, v in clusters.items():
            if len(v) == 0:
                remove.append(k)

        # Remove from the dictionary
        for k in remove:
            del clusters[k]

        return clusters

    # Determine if a point is valid, used by DFS. Valid signifies that it is on the grid, not visited and is holding a
    # datum vector
    def is_valid(self, i, j, visited):
        return 0 <= i < self.grid_size + 1 and 0 <= j < self.grid_size + 1 and not visited[i][j] and \
               self.grid[(i, j)]['Datum'] is not None

    # Recursive DFS algorithm
    def DFS(self, i, j, visited, cluster):
        # The square search space around a point which can be valid for connection. 2 would mean two points left,
        # two right, two down, etc and is generally safe and close to what shows within a graph.
        # 1 creates many small clusters
        threshold = 2

        # Create vectors of valid x, y coordinates which are around the point we're at
        rows = []
        cols = []
        for x in range(-threshold, threshold + 1):
            for y in range(-threshold, threshold + 1):
                if x == 0 and y == 0:
                    continue
                else:
                    rows.append(x)
                    cols.append(y)

        # Set the current point to visited
        visited[i][j] = True

        # Iterate through all reachable points
        for z in range(len(rows)):
            # Determine if the point is valid
            if self.is_valid(i + rows[z], j + cols[z], visited):
                # Append the location to the cluster list, and recure from it
                cluster.append((i + rows[z], j + cols[z]))
                self.DFS(i + rows[z], j + cols[z], visited, cluster)

    # Main ACO algorithm based on the Lumer-Faieta algorithm
    def main(self, name, max_iter=1000000):
        for i in range(max_iter):
            if i % 10000 == 0:
                print("Current iteration:", i)
            # Iterate through all ants
            for ant in self.ants:
                # If the ant is not carrying anything and there is a datum available, try to pick it up
                if ant.carrying is None and self.grid[ant.location]['Datum'] is not None:
                    ant.pickup(self.prob_pick(ant), self.grid)
                # If the ant is carrying a datum and there is a spot available, try to place it on the grid
                elif ant.carrying is not None and self.grid[ant.location]['Datum'] is None:
                    ant.drop(self.prob_drop(ant), self.grid)

                # Find a valid postion for the ant to move to, then update its position
                rand_pos = self.find_valid_pos(ant)
                self.grid[ant.location]['Ant'] = None
                self.grid[rand_pos]['Ant'] = ant
                ant.location = rand_pos

        # Make all the ants drop any datums they happen to be carrying at the end of iterations. They will find the
        # proper location before dropping.
        while len(self.ants) > 0:
            for ant in self.ants:
                if ant.carrying is None:
                    self.ants.remove(ant)

                # Try to drop the point
                if ant.carrying is not None and self.grid[ant.location]['Datum'] is None:
                    ant.drop(self.prob_drop(ant), self.grid)

                # Find a valid postion for the ant to move to, then update its position
                rand_pos = self.find_valid_pos(ant)
                self.grid[ant.location]['Ant'] = None
                self.grid[rand_pos]['Ant'] = ant
                ant.location = rand_pos

        # Find the clusters using helper functions
        clusters_points, clusters_real = self.find_clusters()
        self.graph_grid(name, clusters_points)
        return clusters_real


if __name__ == '__main__':
    aco = ACO(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    aco.main(100000)
