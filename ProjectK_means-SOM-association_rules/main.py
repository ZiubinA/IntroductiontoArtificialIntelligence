import numpy as np
import pandas as pd

df = pd.read_csv('car_price_prediction_with_missing.csv')

if 'Car ID' in df.columns:
    df = df.drop('Car ID', axis=1)

numeric_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Standardize data (Z-score)
X = df_encoded.values.astype(float)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1 
X_scaled = (X - X_mean) / X_std

# SOM
class SelfOrganizingMap:
    def __init__(self, width, height, input_dim, data, learning_rate=0.1, radius=None):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(width, height) / 2

        # weight initialization to match the scaled input space
        random_indices = np.random.choice(data.shape[0], size=width * height, replace=True)
        self.weights = data[random_indices].reshape(width, height, input_dim)
        
    def _get_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
        
    def train(self, data, num_iterations):
        time_constant = num_iterations / np.log(self.radius)
        
        for i in range(num_iterations):
            r = self.radius * np.exp(-i / time_constant)
            lr = self.learning_rate * np.exp(-i / num_iterations)
            
            sample = data[np.random.randint(0, data.shape[0])]

            bmu_idx = self._get_bmu(sample)
            
            for x in range(self.width):
                for y in range(self.height):
                    w_dist = np.square(x - bmu_idx[0]) + np.square(y - bmu_idx[1])
                    
                    if w_dist <= np.square(r):
                        influence = np.exp(-w_dist / (2 * np.square(r)))
                        self.weights[x, y, :] += lr * influence * (sample - self.weights[x, y, :])

print("starting SOM ")
# Passed X_scaled into the data parameter for proper initialization
som = SelfOrganizingMap(width=10, height=10, input_dim=X_scaled.shape[1], data=X_scaled, learning_rate=0.5)
som.train(X_scaled, num_iterations=1000)

print("learning SOM ended")
print("weights:", som.weights.shape)



# K-MEANS CLUSTERING
class KMeans:
    def __init__(self, k=5, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        np.random.seed(42)
        # Randomly initialize centroids by picking k random data points
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
        labels = np.zeros(X.shape[0])
        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

            labels = np.argmin(distances, axis=1)
            
            # Calculate new centroids
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                points = X[labels == j]
                if len(points) > 0:
                    new_centroids[j] = np.mean(points, axis=0)
                else:
                    new_centroids[j] = X[np.random.choice(X.shape[0])]
                    
            # Check for convergence 
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                print(f"K-Means converged after {i+1} iterations.")
                break
                
            self.centroids = new_centroids
            
        return labels

print("\nstarting K-Means ")
kmeans = KMeans(k=5, max_iters=100)
cluster_labels = kmeans.fit(X_scaled)
print("learning K-Means ended")
print("first 10 cluster labels:", cluster_labels[:10])