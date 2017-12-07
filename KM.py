def train(self):
    training_data = self.training_data
    hidden_layer = self.hidden_layer
    clusters = self.clusters

    converged = False
    while not converged:
        converged = True
        # initialize clusters
        for i in range(0, len(training_data) - 1):
            temp_index = None
            temp_min = None
            for j in range(len(hidden_layer) - 1):
                # to not include outputs in distance calculation
                # if temp_min == None or np.linalg.norm(training_data[i][:len(training_data[i]) -1] - hidden_layer[j][:len(hidden_layer[j]) - 1]
                if temp_min == None or np.linalg.norm(np.array(training_data[i]) - np.array(hidden_layer[j].center)) < temp_min:
                    temp_min =  np.linalg.norm(np.array(training_data[i]) - np.array(hidden_layer[j].center))
                    temp_index = j
            clusters[temp_index].append(i)

        # Calculates new centers
        average = [0] * len(clusters[0])
        for i in range(len(clusters) - 1):
            for item in average:
                item = 0
            if i == 1:
                print(average)
            # For every index in cluster
            for j in range(len(clusters[i]) - 1):
                average = [x + y for x, y in zip(average, training_data[clusters[i][j]])]

            average = [x/len(clusters[i]) for x in average]
            if average != hidden_layer[i].center:
                converged = False
                hidden_layer[i].center = average
