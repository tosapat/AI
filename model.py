import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X_A, _ = make_blobs(n_samples=100, 
                    n_features=2, 
                    centers=[[2.0, 2.0]],    
                    cluster_std=0.75,        
                    random_state=42)

X_B, _ = make_blobs(n_samples=100, 
                    n_features=2, 
                    centers=[[3.0, 3.0]],   
                    cluster_std=0.75,        
                    random_state=42)

X = np.vstack((X_A, X_B))
y = np.hstack((np.ones(100), np.zeros(100)))  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(5,), activation='identity', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

x1_range = np.linspace(0, 5, 500)  
x2_range = np.linspace(0, 5, 500)  
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_points_scaled = scaler.transform(grid_points)  
predictions = model.predict(grid_points_scaled).reshape(x1_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, predictions, levels=[-np.inf, 0.5, np.inf], colors=['blue', 'red'], alpha=0.5)
plt.scatter(X_B[:, 0], X_B[:, 1], c='blue', s=50, label='Class 2')  
plt.scatter(X_A[:, 0], X_A[:, 1], c='red', s=50, label='Class 1')  
plt.contour(x1_grid, x2_grid, predictions, levels=[0], colors='black', linewidths=2)
plt.xlim(0, 5)  
plt.ylim(0, 5) 
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
