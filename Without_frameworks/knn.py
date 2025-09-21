# Coding K-Nearest Neighbors from scratch in pure Python
def euclid_L2(x1, x2):
    return (sum((a - b) ** 2 for a, b in zip(x1, x2))) ** 0.5

class KNN:
    def __init__(self, k):
        self.k = k
        self.x_train = []  # stores the train data
        self.y_train = []

    def predict(self, q):
        # where q is the query point
        distances = []
        for i, x in enumerate(self.x_train):
            d = euclid_L2(x, q)
            distances.append((d, self.y_train[i]))
        distances.sort(key=lambda dist: dist[0])

        # now pick the k-nearest neighbours
        kn = distances[: self.k]

        # counting votes
        hash_map = {}
        for dist, label in kn:
            weights = (1/(dist+1e-9))
            hash_map[label] = hash_map.get(label, 0) + weights
        #using weights will give more accurate results as point near to query gets more weight

        # Find the label with the maximum votes
        return max(hash_map.items(), key=lambda item: item[1])[0]

    def predict_multi(self, xt):#for multiple predictions at a time
        return [self.predict(x) for x in xt]
    
    
##LEts try it out for a sample dataset generated from llm
def run_demo():
    knn = KNN(k=3)
    knn.x_train = [
        [0, 0],   # Red
        [10, 0],  # Blue
        [10, 1],  # Blue
    ]
    knn.y_train = ["red", "blue", "blue"]
    queries = [[1, 0],[8,6]]
    for q in queries:
        print(q, "->", knn.predict(q))
    print("Multi:",knn.predict_multi(queries))

if __name__ == "__main__":
    run_demo()
