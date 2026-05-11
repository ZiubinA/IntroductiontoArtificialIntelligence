import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load the dataset
df = pd.read_csv("heart.csv")
X = df.drop('target', axis=1)
y = df['target']

# Set up 10-fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

print("--- BASELINE MODEL ---")
# 2. Build the Baseline ANN Model
# Architecture: 1 Input layer (13 features), 1 Hidden layer (8 neurons), 1 Output layer
baseline_model = MLPClassifier(
    hidden_layer_sizes=(8,), 
    activation='logistic',     # Sigmoid activation
    learning_rate_init=0.001, 
    max_iter=1000, 
    random_state=42
)

# 3. Apply 10-fold CV on Baseline
baseline_scores = cross_val_score(baseline_model, X, y, cv=kf)
baseline_avg = np.mean(baseline_scores)

print(f"Accuracy at each fold: {np.round(baseline_scores * 100, 2)}")
print(f"Averaged Performance: {baseline_avg * 100:.2f}%\n")


print("--- IMPROVED MODEL ---")
# 4. Build the Improved ANN Model
# Improvements: Data Scaling, Deeper Architecture, ReLU activation, Faster learning rate
improved_model = make_pipeline(
    StandardScaler(), # Rearranges/Normalizes the input dataset 
    MLPClassifier(
        hidden_layer_sizes=(16, 8), # 2 Hidden layers (16 neurons, then 8 neurons)
        activation='relu',          # Changed activation function
        learning_rate_init=0.01,    # Increased learning rate
        max_iter=1000, 
        random_state=42
    )
)

# 5. Apply 10-fold CV on Improved Model
improved_scores = cross_val_score(improved_model, X, y, cv=kf)
improved_avg = np.mean(improved_scores)

print(f"Accuracy at each fold: {np.round(improved_scores * 100, 2)}")
print(f"Averaged Performance: {improved_avg * 100:.2f}%")
print(f"\nTotal Improvement: {(improved_avg - baseline_avg) * 100:.2f}%")