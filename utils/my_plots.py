import matplotlib.pyplot as plt
import numpy as np


def plot_clf_train_scatter(X, y, clf,
                           x_label='X[0]',
                           y_label='X[1]',
                           title='Training results',
                           save_name='clf_model.png'):

    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_range = x.max() - x.min()
        x_min, x_max = x.min() - x_range * 0.1, x.max() + x_range * 0.1
        y_range = y.max() - y.min()
        x_min, x_max = y.min() - y_range * 0.1, y.max() + y_range * 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plt.figure()
    plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(save_name)


def plot_reg_train_scatter(X, y, clf,
                           x_label='x',
                           y_label='y',
                           title='Training results',
                           save_name='reg_model.png'):

    def make_grid(x, n_points=100):
        x_range = x.max() - x.min()
        x_min, x_max = x.min() - x_range * 0.1, x.max() + x_range * 0.1
        xx = np.arange(x_min, x_max, (x_max - x_min)/n_points)
        xx = list(map(lambda el: [el], xx))
        return xx

    def plot_regression(ax, clf, X, alpha=0.8):
        prediction = clf.predict(X)
        ax.plot(X, prediction, color='orange', linewidth=3, alpha=alpha)

    X0 = X[:, 0]
    X_grid = make_grid(X0)

    plt.figure()
    plot_regression(plt, clf, X_grid)
    plt.scatter(X0, y, s=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_name)
