import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

include_svm = False

# Create some artificial data
N_half = 1000
N = N_half * 2
y = np.concatenate((np.ones(shape=(N_half,)), -1 * np.ones(shape=(N_half,))), axis=0)
X = 0.3 * np.random.randn(N, 2)
X[:, 0] += y

num_plots = 3 if include_svm else 2
f, axarr = plt.subplots(num_plots)
plot_extent = [-5, 5, -3, 3]

# Make a grid to evaluate our model on
num_grid = 50
X1_grid, X2_grid = np.meshgrid(np.linspace(-5, 5, num_grid), np.linspace(-3, 3, num_grid))
X_grid = np.stack((X1_grid.flatten(), X2_grid.flatten()), axis=1)

"""
Logistic regression
"""
# Fit a logistic regression model to it
model = LogisticRegression(n_jobs=8, C=10**-2)
model.fit(X, y)
assert np.mean(np.equal(model.predict(X), y)) > 0.9, "Training error for Logistic Regression is below 0.9"

# Evaluate the model on the grid
Y_grid = model.predict_proba(X_grid)[:, 1]
Y_grid_2D = np.reshape(Y_grid, newshape=(num_grid, num_grid))

# All the pyplot magic
ax0 = axarr[0].imshow(Y_grid_2D, extent=plot_extent, cmap="binary")
axarr[0].set_title("Logistic regression model")
f.colorbar(ax0, ax=axarr[0])

"""
Gaussian Process Classifier
"""
# Repeat for the Gaussian Process Classifier
model = GaussianProcessClassifier()
model.fit(X, y)
assert np.mean(np.equal(model.predict(X), y)) > 0.9, "Training error for Logistic Regression is below 0.9"

# Evaluate the model on the grid
Y_grid = model.predict_proba(X_grid)[:, 1]
Y_grid_2D = np.reshape(Y_grid, newshape=(num_grid, num_grid))

# Again some pyplot magic
ax1 = axarr[1].imshow(Y_grid_2D, extent=plot_extent, cmap="binary")
axarr[1].set_title("Gaussian Process model")
f.colorbar(ax1, ax=axarr[1])

"""
Support vector machine
"""
if include_svm:
    # Repeat for the support vector machine
    model = SVC(probability=True)
    model.fit(X, y)
    assert np.mean(np.equal(model.predict(X), y)) > 0.9, "Training error for Logistic Regression is below 0.9"

    # Evaluate the model on the grid
    Y_grid = model.predict_proba(X_grid)[:, 1]
    Y_grid_2D = np.reshape(Y_grid, newshape=(num_grid, num_grid))

    # Again some pyplot magic
    ax2 = axarr[2].imshow(Y_grid_2D, extent=plot_extent, cmap="binary")
    axarr[2].set_title("Support vector machine")
    f.colorbar(ax2, ax=axarr[2])


for ax in axarr:
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="winter", s=5, marker="*")
    ax.set_xlim(plot_extent[:2])
    ax.set_ylim(plot_extent[2:])

plt.show()
