'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''

from collections import OrderedDict
import math
import random


class Ant():
    def __init__(self, location):
        self.location = location
        self.carrying = None

    def move(self, location):
        self.location = location

    def pickup(self, prob, grid):
        if random.random() <= prob:
            self.carrying = grid[self.location]['Datum']
            grid[self.location]['Datum'] = None

    def drop(self, prob, grid):
        if random.random() <= prob:
            grid[self.location]['Datum'] = self.carrying
            self.carrying = None


class ACO():
    def __init__(self, data, patch_size=1, step_size=1, gammas=[0.1, 0.01, 0.5]):
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

    def density(self, ant, pickup):
        min_x = max(0, ant.location[0] - self.patch_size)
        max_x = min(self.grid_size, ant.location[0] + self.patch_size)
        min_y = max(0, ant.location[1] - self.patch_size)
        max_y = min(self.grid_size, ant.location[1] + self.patch_size)
        density_sum = 0

        # Search in the space around the ant
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.grid[(x, y)]['Datum'] is not None:
                    if pickup:
                        density_sum += (1 - (
                                self.similarity(self.grid[ant.location]['Datum'], self.grid[(x, y)]['Datum']) /
                                self.gammas[0]))
                    else:
                        density_sum += (1 - (
                                self.similarity(ant.carrying, self.grid[(x, y)]['Datum']) / self.gammas[0]))

        return max(0, ((1 / self.patch_size ** 2) * density_sum))

    def prob_pick(self, ant):
        return (self.gammas[1] / (self.gammas[1] + self.density(ant, True))) ** 2

    def prob_drop(self, ant):
        density = self.density(ant, False)
        if density < self.gammas[2]:
            return 2 * density
        else:
            return 1

    def eval_clusters(self):
        for k, v in self.grid.items():
            if v['Datum'] is not None:
                print(k)

    def find_valid_pos(self, ant):
        min_x = max(0, ant.location[0] - self.step_size)
        max_x = min(self.grid_size, ant.location[0] + self.step_size)
        min_y = max(0, ant.location[1] - self.step_size)
        max_y = min(self.grid_size, ant.location[1] + self.step_size)

        # Move the ant to an available spot within step size
        rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
        # Loop until we have a point in the grid which isn't taken
        while self.grid[rand_pos]['Ant'] is not None:
            rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))

        return rand_pos

    def main(self, max_iter=10000):
        for i in range(max_iter):
            for ant in self.ants:
                # If the ant is not carrying anything and there is a datum available, try to pick it up
                if ant.carrying is None and self.grid[ant.location]['Datum'] is not None:
                    ant.pickup(self.prob_pick(ant), self.grid)
                # If the ant is carrying a datum and there is a spot available, try to place it on the grid
                elif ant.carrying is not None and self.grid[ant.location]['Datum'] is None:
                    ant.drop(self.prob_drop(ant), self.grid)

                rand_pos = self.find_valid_pos(ant)

                self.grid[ant.location]['Ant'] = None
                self.grid[rand_pos]['Ant'] = ant
                ant.location = rand_pos

        self.eval_clusters()


if __name__ == '__main__':
    aco = ACO(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    aco.main(1000000)
