# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Load the customer dataset and select required features.

2.Choose the number of clusters K and initialize the K-Means model.

3.Fit the model to the data and assign customers to clusters.

4.Display the clustered customer segments and centroids.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: KALPANA M
RegisterNumber: 212225240064 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())
X = data.iloc[:, [3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], 
kmeans.cluster_centers_[:,1], 
s=300, c='yellow', label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Output:
<img width="1204" height="128" alt="Screenshot 2026-05-18 141355" src="https://github.com/user-attachments/assets/a3484cfc-901f-4951-971e-f2836a80cbe9" />
<img width="722" height="468" alt="download" src="https://github.com/user-attachments/assets/a5ca1254-96ed-4044-87fe-141b6dc180da" />
<img width="695" height="545" alt="download" src="https://github.com/user-attachments/assets/69fe303d-b28d-4171-bbc8-c96a7dfcbdf9" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
