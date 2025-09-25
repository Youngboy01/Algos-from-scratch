# unsupervised learning algorithm
import random


def euclid(x1, x2):
    return (sum((a - b) ** 2 for a, b in zip(x1, x2))) ** 0.5


class K_means:
    def __init__(self, k, n) -> None:
        self.k = k
        self.n = n

    def cluster(self, data):
        k_centroids = random.sample(data, self.k)
        for _ in range(self.n):
            clusters = [[] for _ in range(self.k)]
            for x in data:
                distances = [euclid(x, mu) for mu in k_centroids]
                ci = distances.index(min(distances))
                clusters[ci].append(x)

            # new centroid calculation
            new_centroid = []
            for cluster in clusters:
                if cluster:  
                    centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
                else:
                    centroid = random.choice(data)
                new_centroid.append(centroid)
            #compute loss
            loss=0
            for ci,cluster in enumerate(clusters):
                for x in cluster:
                    loss += euclid(x,new_centroid[ci])**2
            
            if new_centroid==k_centroids:
                print("task completed")
                break
            k_centroids = new_centroid
            
            
        return clusters,k_centroids,loss 
    
    

data = [(1,2), (1,4), (2,3), (8,9), (9,10)]
kmeans = K_means(k=2, n=10)
clusters, centroids, loss = kmeans.cluster(data)

print("\nFinal Clusters:", clusters)
print("Final Centroids:", centroids)
print("Final Loss:", loss)
