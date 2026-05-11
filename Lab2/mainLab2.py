import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

data = np.loadtxt('sunspot.txt')
years = data[:, 0]
sunspots = data[:, 1]

plt.figure(figsize=(12, 5))
plt.plot(years, sunspots, 'r-*')
plt.xlabel('Year') 
plt.ylabel('Number of Sunspots') 
plt.title('Sunspot Activity (1700-2014)') 
plt.grid(True)
plt.show()

def run_autoregressive_model(n, is_iterative=False, learning_rate=1e-5):
    print(f"\n--- Running Model with n={n} ---")
    L = len(sunspots) 
    
    P = np.array([sunspots[i:L-n+i] for i in range(n)]).T
    T = sunspots[n:L]
    
    # 3D Plot 
    if n == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], T, c='b', marker='o')
        ax.set_xlabel('a(k-2)')
        ax.set_ylabel('a(k-1)')
        ax.set_zlabel('a(k)')
        plt.title('Inputs vs Outputs (n=2)')
        plt.show()

    # Select the first 200 members for training 
    Pu = P[:200, :]
    Tu = T[:200]
    
    P_test = P[200:, :]
    T_test = T[200:]

    #Create and train the artificial neuron 
    if not is_iterative:
        # Exact calculation using Matrix Calculus (Matlab's newlind)
        model = LinearRegression()
    else:
        # SGDRegressor acts as an Adaptive Linear Neuron. We adjust eta0 (learning rate).
        model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=1000, tol=1e-3)
        
    model.fit(Pu, Tu)
    
    # Display neuron weight coefficient values 
    weights = model.coef_
    bias = model.intercept_
    print(f"Weights (w): {weights}")
    print(f"Bias (b): {bias}")

    Tsu = model.predict(Pu) # Forecast for training data
    Ts_all = model.predict(P) 
    
    plt.figure(figsize=(12, 5))
    plt.plot(years[n:], T, 'b-', label='True Values (T)')
    plt.plot(years[n:], Ts_all, 'r--', label='Forecasted Values (Ts)')
    plt.xlabel('Year')
    plt.ylabel('Sunspots')
    plt.title(f'True vs Forecasted Sunspots (n={n}, Iterative={is_iterative})')
    plt.legend()
    plt.grid(True)
    plt.show()

    e = T - Ts_all
    
    plt.figure(figsize=(12, 5))
    plt.plot(years[n:], e, 'k-')
    plt.xlabel('Year')
    plt.ylabel('Error')
    plt.title(f'Forecast Error over Time (n={n})') 
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(e, bins=30, edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Forecast Error Histogram')
    plt.show()

    mse = mean_squared_error(T, Ts_all)
    mad = np.median(np.abs(e)) 
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Median Absolute Deviation (MAD): {mad:.4f}")

run_autoregressive_model(n=2, is_iterative=False)

run_autoregressive_model(n=2, is_iterative=True, learning_rate=1e-5)

run_autoregressive_model(n=6, is_iterative=True, learning_rate=1e-6)

run_autoregressive_model(n=11, is_iterative=True, learning_rate=1e-7)