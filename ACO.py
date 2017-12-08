'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''

import random
from collections import OrderedDict


class Ant():
    def __init__(self, location):
        self.location = location
        self.carrying = None

    def move(self, step):
        pass

    def pickup(self):
        pass

    def drop(self):
        pass


class ACO():
    def __init__(self, data, patch_size=1, step_size=1, gammas=[1, 0.01, 0.01]):
        # Find how large the dataset is and determine the grid size and amount of ants based on that
        length = len(data)
        self.grid_size = length * 10
        # OrderedDict where the keys are tuples for coordinates (x, y), and the value is a dictionary
        self.grid = self.init_grid(self.grid_size, data)
        self.ants = self.init_ants(length * 2)
        # The search space of the ants
        self.patch_size = patch_size
        # The amount of spaces an ant can move per move
        self.step_size = step_size
        # Tunable parameters for probabilities
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

    def similarity(self, point):
        pass

    def density(self, ant):
        pass

    def prob_pick(self, ant):
        pass

    def prob_drop(self, ant):
        pass

    def eval_clusters(self):
        pass

    def main(self, max_iter=10000):
        pass


if __name__ == '__main__':
    aco = ACO(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    # print(aco.grid)
