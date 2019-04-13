import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def PCA(X):
    pca = PCA(n_components=2)
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    print(pca.singular_values_)

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X)

    print(pca.explained_variance_ratio_)

    print(pca.singular_values_)


def Isolation_Forest(X_train, X_test):
    # fit the model

    clf = IsolationForest(behaviour='new', max_samples=100, contamination='auto')
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Generate some abnormal novel observations NOT SURE HOW TO TWEAK THIS:
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

    y_pred_outliers = clf.predict(X_outliers)

    # plot the line, the samples, anid the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # CODE FOR PLOTTING
    #
    # plt.title("IsolationForest")
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    #
    # b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
    #                  s=20, edgecolor='k')
    # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
    #                  s=20, edgecolor='k')
    # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
    #                 s=20, edgecolor='k')
    # plt.axis('tight')
    # plt.xlim((-5, 5))
    # plt.ylim((-5, 5))
    # plt.legend([b1, b2, c],
    #            ["training observations",
    #             "new regular observations", "new abnormal observations"],
    #            loc="upper left")
    # plt.show()