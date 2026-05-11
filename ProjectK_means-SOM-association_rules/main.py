import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

#ASSOCIATION RULES 
def get_frequent_itemsets(transactions, min_support=0.1, max_length=2):
    num_transactions = len(transactions)
    item_counts = {}
    
    for transaction in transactions:
        for item in transaction:
            frozen_item = frozenset([item])
            item_counts[frozen_item] = item_counts.get(frozen_item, 0) + 1
            
    # Filter by minimum support
    frequent_itemsets = {itemset: count/num_transactions 
                         for itemset, count in item_counts.items() 
                         if count/num_transactions >= min_support}
    
    current_l = list(frequent_itemsets.keys())
    k = 2
    all_frequent = dict(frequent_itemsets)
    
    #Iteratively find k-itemsets
    while current_l and k <= max_length:
        candidates = set()
        # Create combinations of length k
        for i in range(len(current_l)):
            for j in range(i+1, len(current_l)):
                union_set = current_l[i] | current_l[j]
                if len(union_set) == k:
                    candidates.add(union_set)
        
        candidate_counts = {c: 0 for c in candidates}
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
                    
        current_l = []
        for candidate, count in candidate_counts.items():
            support = count / num_transactions
            if support >= min_support:
                current_l.append(candidate)
                all_frequent[candidate] = support
        k += 1
        
    return all_frequent

def generate_association_rules(frequent_itemsets, min_confidence=0.5):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for item in itemset:
                antecedent = itemset - frozenset([item])
                consequent = frozenset([item])
                
                if antecedent in frequent_itemsets:
                    confidence = support / frequent_itemsets[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence))
    return rules

print("\nstarting Association Rules (Apriori)")

#prepare transactions based on categorical columns 
transactions = []
for _, row in df[cat_cols].iterrows():
    transaction = [f"{col}_{val}" for col, val in row.items()]
    transactions.append(transaction)

#run algorithm
frequent_itemsets = get_frequent_itemsets(transactions, min_support=0.1, max_length=2)
rules = generate_association_rules(frequent_itemsets, min_confidence=0.5)

print("learning Association Rules ended")
print(f"Found {len(frequent_itemsets)} frequent itemsets and {len(rules)} rules.")

# Print the top 5 most confident rules
rules.sort(key=lambda x: x[3], reverse=True)
print("Top 5 Association Rules:")
for i, (antecedent, consequent, support, confidence) in enumerate(rules[:5]):
    ant_str = ', '.join(list(antecedent))
    con_str = list(consequent)[0]
    print(f"  Rule {i+1}: [{ant_str}] -> [{con_str}] | Support: {support:.2f}, Confidence: {confidence:.2f}")


#SOM Visualization: U-Matrix (Distance Map)
print("\nGenerating SOM U-Matrix plot...")
u_matrix = np.zeros((som.width, som.height))
for x in range(som.width):
    for y in range(som.height):
        neighbors = []
        if x > 0: neighbors.append(som.weights[x-1, y, :])
        if x < som.width - 1: neighbors.append(som.weights[x+1, y, :])
        if y > 0: neighbors.append(som.weights[x, y-1, :])
        if y < som.height - 1: neighbors.append(som.weights[x, y+1, :])
        
        distances = [np.linalg.norm(som.weights[x, y, :] - n) for n in neighbors]
        u_matrix[x, y] = np.mean(distances)

plt.figure(figsize=(8, 6))
sns.heatmap(u_matrix, cmap='viridis')
plt.title('Self-Organizing Map: U-Matrix (Neuron Distances)')
plt.xlabel('SOM X-axis')
plt.ylabel('SOM Y-axis')
plt.show()


#K-Means Visualization: Cluster Distribution
print("Generating K-Means Cluster plot...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, edgecolor='k')
plt.title('K-Means Cluster Distribution (PCA Reduced 2D Space)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(kmeans.k)], title="Clusters")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


#Association Rules Visualization: Support vs Confidence
print("Generating Association Rules plot...")
if len(rules) > 0:
    supports = [rule[2] for rule in rules]
    confidences = [rule[3] for rule in rules]

    plt.figure(figsize=(8, 6))
    plt.scatter(supports, confidences, alpha=0.7, color='coral', edgecolor='k', s=50)
    plt.title('Association Rules: Support vs. Confidence')
    plt.xlabel('Support (Frequency)')
    plt.ylabel('Confidence (Reliability)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adding a legend for clarity even in a simple scatter plot
    plt.scatter([], [], color='coral', edgecolor='k', label='Generated Rule')
    plt.legend()
    plt.show()
else:
    print("No rules generated to plot. Try lowering min_support or min_confidence.")