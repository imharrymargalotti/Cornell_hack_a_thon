import numpy as np
import os
from scipy import stats
from Deploy import environset
import numpy as np
from sklearn.decomposition import PCA


environset.set_paths()


def clr(X):
    to_return = np.zeros(X.shape)
    m = X.shape[0]
    for i in range(0,m):
        x_gmean = stats.gmean(X[:,i])
        to_return[:,i] = np.log(X[:,i] / x_gmean)

    return to_return


def read_in_data():
    path = os.environ.get("COW_CSV")

    with open(path) as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        #print(line.replace("\t", ","))
        data.append(line.replace("\t", ","))

    s = []
    for i in range(2, len(data)):
        s.append(data[i].split(","))

    first_col = []
    for item in s:
        first_col.append(item[0])

    info = []
    for i in s:
        nums = []
        for j in range(1, len(i)):
            nums.append(float(i[j]))
        info.append(nums)

    results = np.asarray(info)
    ids = np.asarray(first_col)

    return ids, results


def pca_reduction(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    print("explained variance ratio")
    print(pca.explained_variance_ratio_)
    print("singular values")
    print(pca.singular_values_)


    print("With SVD Solver:")
    pca = PCA(n_components=2, svd_solver='auto')
    pca.fit(X)
    print("explained variance ratio")
    print(pca.explained_variance_ratio_)
    print("singular values")
    print(pca.singular_values_)


def main():
    id, res =read_in_data()
    pca_reduction(res)



main()

