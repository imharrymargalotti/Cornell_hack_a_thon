import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from Jeff.py import *

from Jeff import ProcessData, plotRoc


def RandForest(AllData, y):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    X = AllData[0]
    y_train = y[0]
    clf.fit(X, y_train)  # TRAIN ON YU
    for x in range(3):  # loops case
        case = AllData[x]
        y_predict = clf.predict(case)
        fpr, tpr, thresholds = metrics.roc_curve(y[x], y_predict)
        plotRoc(fpr, tpr, str(x+10)+"_plot.png")


def AnomolyDetector(AllCases, y):
    # xtrain is yu healthy
    # assumes largest set 'to train on' is first
    H, D = ProcessData(AllCases[0], y[0])
    clf = IsolationForest(behaviour='new', max_samples=100, contamination='auto')
    clf.fit(H)
    for x in range(3):  # loops case
        case = AllCases[x]
        h, d = ProcessData(case, y[x])
        pltDist(h, d, clf, (str(x) + "_plot"))


def PooledRF(AllData, y):
    combinedx = AllData[0]
    combinedx = np.concatenate(AllData[1])

    combinedy = y[0]
    combinedy = np.concatenate(y[1])

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(combinedx,combinedy)
    y_predict = clf.predict(AllData[2])
    fpr, tpr, thresholds = metrics.roc_curve(y[2], y_predict)
    plotRoc(fpr,thresholds,"Pooled.png")

def fSig(AllData, y):
    combinedx = AllData[0]
    combinedx = np.concatenate(AllData[1])

    combinedy = y[0]
    combinedy = np.concatenate(y[1])

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(combinedx, combinedy)
    selector = RFE(clf,5,1,0)
    selector.fit(combinedx,combinedy)
    rankedF = selector.ranking_
    return rankedF

#centered logarithmic ratio
# def pca_reduction(X):
#     pca = PCA(n_components=2)
#     pca.fit(X)
#
#     print(pca.explained_variance_ratio_)
#
#     print(pca.singular_values_)
#
#     pca = PCA(n_components=2, svd_solver='full')
#     pca.fit(X)
#
#     print(pca.explained_variance_ratio_)
#
#     print(pca.singular_values_)
#
#
# def Isolation_Forest(X_train, X_test):
#     # fit the model
#
#     clf = IsolationForest(behaviour='new', max_samples=100, contamination='auto')
#     clf.fit(X_train)
#     y_pred_train = clf.predict(X_train)
#     y_pred_test = clf.predict(X_test)
#
#     # Generate some abnormal novel observations NOT SURE HOW TO TWEAK THIS:
#     X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
#
#     y_pred_outliers = clf.predict(X_outliers)
#
#     # plot the line, the samples, anid the nearest vectors to the plane
#     xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     # CODE FOR PLOTTING
#     #
#     # plt.title("IsolationForest")
#     # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#     #
#     # b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
#     #                  s=20, edgecolor='k')
#     # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
#     #                  s=20, edgecolor='k')
#     # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
#     #                 s=20, edgecolor='k')
#     # plt.axis('tight')
#     # plt.xlim((-5, 5))
#     # plt.ylim((-5, 5))
#     # plt.legend([b1, b2, c],
#     #            ["training observations",
#     #             "new regular observations", "new abnormal observations"],
#     #            loc="upper left")
#     # plt.show()

