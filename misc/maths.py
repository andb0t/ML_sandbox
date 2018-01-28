import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll

print('Load the 3d dataset')
X, t = make_swiss_roll(n_samples=100, noise=0, random_state=1337)

y = X[:, 0] < 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], zdir='z', c=y, cmap=plt.cm.coolwarm)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Swiss roll example')
plt.savefig("3d_dataset.png")

print('Do PCA')
PCA_strategy = 'sklearn'
if PCA_strategy == 'by_hand':
    print('Center the dataset')
    X_centered = X - X.mean(axis=0)
    print('X')
    print(X)
    print('X_centered')
    print(X_centered)
    print('Do the SVD')
    U, s, V = np.linalg.svd(X_centered)
    c1 = V.T[:, 0]
    c2 = V.T[:, 1]
    print('U')
    print(U)
    print('s')
    print(s)
    print('V')
    print(V)
    print('c1')
    print(c1)
    print('c2')
    print(c2)
    print('Project on plane of first two PCs')
    W2 = V.T[:, :2]
    X2D = X_centered.dot(W2)
    print('W2')
    print(W2)
    print('X2D')
    print(X2D)
elif PCA_strategy == 'sklearn':
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)
    print('X2D')
    print(X2D)
