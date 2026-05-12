import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
print("Generating K-Means Cluster plot (with Manual PCA)...")

# Manual PCA implementation
cov_matrix = np.cov(X_scaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvectors by eigenvalues in descending order
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Select the top 2 principal components
eigenvector_2d = sorted_eigenvectors[:, 0:2]

# Project the data
X_pca = np.dot(X_scaled, eigenvector_2d)

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

# 1. DATA PREPARATION
df = pd.read_csv('car_price_prediction_with_missing.csv')
if 'Car ID' in df.columns:
    df = df.drop('Car ID', axis=1)

target_col = 'Price' # Change this if your target column has a different name

# Impute target before extracting it
if df[target_col].isnull().sum() > 0:
    df[target_col] = df[target_col].fillna(df[target_col].median())

y_raw = df[target_col].values
X_raw = df.drop(target_col, axis=1)

# Handle missing values
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
cat_cols = X_raw.select_dtypes(include=['object']).columns

for col in numeric_cols:
    X_raw[col] = X_raw[col].fillna(X_raw[col].median())
for col in cat_cols:
    X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0])

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

# Standardize Features (X)
X = X_encoded.values.astype(float)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1 
X_scaled = (X - X_mean) / X_std

# Standardize Target (y)
y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
if y_std == 0:
    y_std = 1  # prevent divide-by-zero
y_scaled = ((y_raw - y_mean) / y_std).reshape(-1, 1)



# ==========================================
# 2. MANUAL ARTIFICIAL NEURAL NETWORK (MLP) - BULLETPROOF VERSION
# ==========================================
class ScratchANN:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.001):
        self.lr = learning_rate
        self.loss_history = []
        
        # FIX 1: "He Initialization" (Much safer for ReLU networks)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2
        return self.A2

    def backward(self, X, y, output):
        m = X.shape[0]
        
        dZ2 = output - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # FIX 2: Gradient Clipping (Forces the math to stay stable)
        clip_value = 5.0
        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        self.loss_history = [] 
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            loss = np.mean(np.square(output - y))
            # Explicitly save as standard float to prevent plotting errors
            self.loss_history.append(float(loss))
            
    def predict(self, X):
        return self.forward(X)


# ==========================================
# 3. MANUAL CROSS-VALIDATION FUNCTION
# ==========================================
def manual_cross_validation(X, y, hidden_size, learning_rate, epochs, k=5):
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Split indices into k folds
    folds = np.array_split(indices, k)
    rmse_scores = []
    
    for i in range(k):
        # Create train and test sets for this fold
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Initialize and train model
        model = ScratchANN(input_size=X.shape[1], hidden_size=hidden_size, learning_rate=learning_rate)
        model.train(X_train, y_train, epochs=epochs)
        
        # Predict and calculate RMSE on test set
        predictions = model.predict(X_test)
        mse = np.mean(np.square(predictions - y_test))
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)
        
    return rmse_scores


# ==========================================
# 4. EXECUTION: INITIAL VS UPDATED (FIXED LEARNING RATES)
# ==========================================
print("\n--- 1. Initial ANN (Manual Implementation) ---")
# Lowered learning rate drastically to prevent exploding gradients
init_scores = manual_cross_validation(X_scaled, y_scaled, hidden_size=10, learning_rate=0.0001, epochs=500)
print(f"Initial CV RMSE Scores: {[round(score, 4) for score in init_scores]}")
print(f"Average Initial RMSE: {np.mean(init_scores):.4f} (Scaled Units)\n")

# Train once on full data to get the cost function curve
ann_initial = ScratchANN(input_size=X_scaled.shape[1], hidden_size=10, learning_rate=0.0001)
ann_initial.train(X_scaled, y_scaled, epochs=500)

print("--- 2. Undertaken Changes ---")
print("Architecture Update: Increased hidden layer size from 10 to 30 neurons.")
print("Hyperparameter Update: Increased learning rate to 0.0005 and epochs to 1000 for deeper convergence.\n")

print("--- 3. Updated ANN (Manual Implementation) ---")
updated_scores = manual_cross_validation(X_scaled, y_scaled, hidden_size=30, learning_rate=0.0005, epochs=1000)
print(f"Updated CV RMSE Scores: {[round(score, 4) for score in updated_scores]}")
print(f"Average Updated RMSE: {np.mean(updated_scores):.4f} (Scaled Units)\n")

ann_updated = ScratchANN(input_size=X_scaled.shape[1], hidden_size=30, learning_rate=0.0005)
ann_updated.train(X_scaled, y_scaled, epochs=1000)

# ==========================================
# 5. VISUALIZATION: COST FUNCTION
# ==========================================
print("Generating Cost Function Plot...")
plt.figure(figsize=(10, 6))
# Updated the labels to reflect the correct, stable learning rates
plt.plot(ann_initial.loss_history, label='Initial ANN (10 neurons, lr=0.0001)', linestyle='--')
plt.plot(ann_updated.loss_history, label='Updated ANN (30 neurons, lr=0.0005)', linewidth=2)
plt.title('Manual ANN: Cost Function Optimization (MSE) Over Epochs')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Cost (Mean Squared Error)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

