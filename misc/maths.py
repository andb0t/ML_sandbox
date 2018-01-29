import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll


print('Load the 3d dataset')
X, color = make_swiss_roll(n_samples=200, noise=0, random_state=1337)

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

PCA_strategy = 'kernel_pipeline'

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
elif PCA_strategy == '2comp':
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2D = pca.fit(X)
    c1 = pca.components_.T[:, 0]
    c2 = pca.components_.T[:, 1]
    print('c1')
    print(c1)
    print('c2')
    print(c2)
    X2D = pca.transform(X)
    print('X2D')
    print(X2D)
    print('explained variance ratio:', pca.explained_variance_ratio_)

    plt.figure()
    plt.scatter(X2D[:, 0], X2D[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.savefig('x_reduced.png')

elif PCA_strategy == 'cumsum':
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum > 0.95) + 1

    print('Optimal dimensionality', d)
    plt.figure()
    plt.scatter(list(range(1, len(cumsum) + 1)), cumsum)
    plt.xlabel('Dimension')
    plt.ylabel('Explained variance')
    plt.savefig('expl_var_ratio.png')

    X_reduced = pca.tansform(X)
    print(X_reduced)

elif PCA_strategy == '0p95_var':
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print(X_reduced)

elif PCA_strategy == 'incremental':
    # alternatively use mp.memmap for loading file in batches
    from sklearn.decomposition import IncrementalPCA

    n_batches = 5
    inc_pca = IncrementalPCA(n_components=2)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)
    X2D = inc_pca.transform(X)
    print(D2D)

elif PCA_strategy == 'randomized':
    from sklearn.decomposition import PCA

    rnd_pca = PCA(n_components=2, svd_solver='randomized')
    X_reduced = rnd_pca.fit_transform(X)
    print(X_reduced)

elif PCA_strategy == 'KernelPCA':
    from sklearn.decomposition import KernelPCA

    rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
    X2D = rbf_pca.fit_transform(X)

    plt.figure()
    plt.scatter(X2D[:, 0], X2D[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.savefig('x_reduced_rbf_kernel.png')

elif PCA_strategy == 'kernel_pipeline':
    from sklearn.decomposition import KernelPCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ('kpca', KernelPCA(n_components=2)),
        ('log_reg', LogisticRegression())
        ])

    param_grid = [{
        # 'kpca__n_components': [1, 2, 3],
        'kpca__gamma': np.linspace(0.03, 0.05, 10),
        'kpca__kernel': ['rbf', 'sigmoid']
        }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)

    print(grid_search.best_params_)

    print('Use optimal keys in PCA')
    kpca_par_dict = grid_search.best_params_
    best_keys = list(grid_search.best_params_.keys())
    best_keys_clean = map(lambda x: x.split('kpca__')[1], best_keys)
    for idx, key in enumerate(best_keys_clean):
        kpca_par_dict[key] = kpca_par_dict.pop(best_keys[idx])
    print(kpca_par_dict)
    kpca = KernelPCA(n_components=2, **kpca_par_dict)
    X2D = kpca.fit_transform(X)
    #
    plt.figure()
    plt.scatter(X2D[:, 0], X2D[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.savefig('x_reduced_kernel_pipeline.png')

elif PCA_strategy == 'LLE':
    from sklearn.manifold import LocallyLinearEmbedding

    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    X2D = lle.fit_transform(X)

    plt.figure()
    plt.scatter(X2D[:, 0], X2D[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.savefig('x_reduced_LLE.png')
