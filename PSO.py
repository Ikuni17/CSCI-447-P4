import random
import numpy as np
import KM
import experiment


class Particle:
    gbest = None
    gscore = None
    v1 = None
    v2 = None

    def __init__(self, clusters, data):
        self.pbest = None
        self.pscore = None
        self.centers_pos = [] # holds the centers as [[x1,y1],[x2,y2]]
        self.velocity = [] # holds the current velocity as [[x1,y1],[x2,y2]]

        # Initialize random centers and velocity arrays
        for i in range(clusters):
            self.centers_pos.append(list(data[random.randint(0, len(data) - 1)]))
            self.velocity.append([])
            for j in range(len(data[0])):
                self.velocity[i].append(0)
        self.centers_pos = self.centers_pos

    # calculates the average distance to center for each cluster
    def evaluate(self, data):
        # calculates lists of clustered points based on the centers
        clusters = Particle.get_clusters(self.centers_pos, data)
        centers = []
        for cluster in clusters:
            if (cluster):
                center_point = []

                # Calculate the average in each dimension for the cluster
                for i in range(len(data[0])):
                    dimCut = [dim[i] for dim in cluster] # Get all the values for a dimension in the cluster
                    center_point.append(sum(dimCut) / len(dimCut)) # Calculate the average of that dimension
                centers.append(center_point)

            dist = 0
            # Calculate the distance from each point to its center
            for point in cluster:
                dist = dist + KM.euclidian_distance(point, center_point)

        average = dist / len(data)
        return average

    # Find the closest center for a point and assign it to that cluster
    def get_clusters(centers, data):
        # Initialize clusters with the right length
        clusters = []
        for i in range(len(centers)):
            clusters.append([])

        # Find each points closest center
        for point in data:
            closest = 0
            for i in range(len(centers)):
                if abs(KM.euclidian_distance(centers[i], point)) < abs(KM.euclidian_distance(centers[closest], point)):
                    closest = i
            # Assign it to a point
            clusters[closest].append(point)
        return clusters

    def update_velocity(self):
        # For every center's velocity
        for i in range(len(self.velocity)):
            # For every dimension in the data
            for j in range(len(self.velocity[i])):
                # Calculate the new velocity based on the update rule
                self.velocity[i][j] = self.velocity[i][j] + Particle.v1 * random.random() * (
                        self.pbest[i][j] - self.centers_pos[i][j]) + Particle.v2 * random.random() * (
                                              Particle.gbest[i][j] - self.centers_pos[i][j])

    def move(self, data):
        self.update_velocity()

        # For evey center
        for i in range(len(self.centers_pos)):
            # For every dimension
            for j in range(len(self.centers_pos[i])):
                # upadate the position
                self.centers_pos[i][j] = self.centers_pos[i][j] + self.velocity[i][j]
        # Evaluate the position
        performance = self.evaluate(data)

        # update personal and/or global bests if we found a better position
        if performance < Particle.gscore:
            Particle.gbest = self.centers_pos
            self.pbest = self.centers_pos
        elif performance < self.pscore:
            self.pbest = self.centers_pos


class PSO:
    def __init__(self, numParticles, clusters, data, v1=0.01, v2=0.01):
        # initialize the particles with random values
        self.data = data
        self.particles = self.initSwarm(numParticles, clusters, data, v1, v2)

    # Move each particle around for the number of rounds and return the clusters as lists
    def runSwarm(self, rounds):
        for i in range(rounds):
            if i % 10 == 0:
                print("PSO, current iteration:", i)
            for particle in self.particles:
                particle.move(self.data)
        print('calculaing clusters')
        return Particle.get_clusters(Particle.gbest, self.data)

    # Initialize particles and return them as a list
    def initSwarm(self, numParticles, clusters, variables, v1, v2):
        particles = []
        for i in range(numParticles):
            particles.append(Particle(clusters, variables))
            particles[i].pbest = particles[i].centers_pos
            particles[i].pscore = particles[i].evaluate(self.data)
        Particle.gbest = particles[0].pbest
        Particle.gscore = particles[0].pscore
        Particle.v1 = v1
        Particle.v2 = v2
        return particles


if __name__ == '__main__':
    data_name = 'datasets/bean.csv'
    iterations = 100
    data = KM.read_data(data_name)
    clusters = 5
    particles = 10
    pso = PSO(particles, clusters, data)
    print('Running PSO on {0} with {1} clusters and {2} particles for {3} iterations'.format(data_name, clusters,
                                                                                             particles, iterations))
    best = pso.runSwarm(iterations)

    # print(experiment.evaluate_clusters(best))
    experiment.graph2dClusters(best)
