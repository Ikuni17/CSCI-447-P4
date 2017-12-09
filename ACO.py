'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''

from collections import OrderedDict
import math
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

    def eval_clusters(self):
        for k, v in self.grid.items():
            if v['Datum'] is not None:
                print(k)

    # Find a valid position for an ant to move to within step size
    def find_valid_pos(self, ant):
        # Only try positions which
        min_x = max(0, ant.location[0] - self.step_size)
        max_x = min(self.grid_size, ant.location[0] + self.step_size)
        min_y = max(0, ant.location[1] - self.step_size)
        max_y = min(self.grid_size, ant.location[1] + self.step_size)
        # Fail safe if an ant is stuck, i.e. its surrounded by ants and cannot move
        i = 0

        # Move the ant to an available spot within step size
        rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
        # Loop until we have a point in the grid which isn't taken
        while self.grid[rand_pos]['Ant'] is not None and i <= (self.step_size * 1000):
            rand_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            i += 1

        # If we reached the fail safe, just leave the ant and hope another moves before the next iteration
        if i == (self.step_size * 1000):
            return ant.location
        else:
            return rand_pos

    # Main ACO algorithm based on the Lumer-Faieta algorithm
    def main(self, max_iter=10000):
        for i in range(max_iter):
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

        self.eval_clusters()


if __name__ == '__main__':
    aco = ACO(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    aco.main(1000000)
