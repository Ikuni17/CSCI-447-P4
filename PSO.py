import random
import numpy as np

class Particle:
    gbest = None
    v1 = None
    v2 = None

    def __init__(self, clusters, variables):
        self.pbest = None
        self.clusterPositions = []
        self.velocity = []
        for i in range(clusters):
            self.clusterPositions.append([])
            self.velocity.appdn([])
            for j in range(variables):
                self.clusterPositions[i].append(random.randint(0, 30))
                self.velocity.append(0)
        self.clusterPositions = np.array(self.clusterPositions)
        self.evaluate()

    # calculate the distance of the clusters and update gbest and pbest
    def evaluate(self):
        pass

    def calcVelocity(self):
        for i in range(len(velocity)):
            for j in range(len(velocity[i])):
                # The velocity equation
                self.velocity[i][j] = self.velocity[i][j] + v1*random.uniform(0, 1)*(self.pbest[i][j] - self.clusterPosition[i][j]) + Particle.v2*random.uniform(0, 1)*(Particle.gbest[i][j] - self.clusterPosition[i][j])
        return self.velocity

    def updatePosition(self):
        np.add(self.clusterPositions, calcVelocity())
        self.evaluate()

class PSO:
    def __init__(self, numParticles, clusters, data, v1 = 0.1, v2 = 0.1):
        #initialize the particles with random values
        particles = self.initSwarm(numParticles, clusters, len(data[0]), v1, v2)

    def initSwarm(self, numParticles, clusters, variables, v1, v2):
        particles = []
        for i in range(numParticles):
            particles.append(Particle(clusters, variables))
        Particle.v1 = v1
        Particle.v2 = v2
        return particles

    def run(rounds):
        # Move the particles around while updating their position
        for i in range(rounds):
            for particle in particles:
                particle.updatePosition()

if __name__ == '__main__':
    data = [[1,5, 7], [5,19, 12], [14, 4, 7], [5, 18, 21], [9,41, 25], [13, 32, 15]]
    PSO(10, 2, data)
