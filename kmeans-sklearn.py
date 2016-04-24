import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

products = range(10000)
users = range(1000)
purchases = []
for p in range(100000):
	u = random.choice(users)
	p = random.choice(products)
	purchases.append([u,p])

X = np.array(purchases);

kmeans = KMeans(n_clusters=8)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y.","m.","k.","b.","w.",]

for i in range(len(X)):
    #print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
