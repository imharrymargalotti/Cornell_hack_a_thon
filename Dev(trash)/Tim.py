import numpy as np
import os
import sklearn
from scipy import stats
from Deploy import environset

environset.set_paths()


def read_in_COW():
    path = os.environ.get("COW_CSV")

    with open(path) as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        print(line.replace("\t", ","))
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


def read_in_ERR():
    path = os.environ.get("ERR")
    input = []
    with open(path) as f:
        lines=f.readlines()
    f.close()
    clean = [x.strip() for x in lines]

    headers = []
    for i in range(1, len(clean)):
        headers.append(clean[i].split(",")[0])

    rows = []
    for i in range(1,len(clean)):
        k = clean[i].split(",")
        nums = []
        for j in range(1, len(k)):
            nums.append(float(k[j]))
        rows.append(nums)

    col = np.asarray(headers)
    data = np.asarray(rows)

    return col, data


def read_in_y():
    path = os.environ.get("Y")
    with open(path) as f:
        lines = f.readlines()
    f.close()
    clean = [int(x.strip()) for x in lines]
    res = np.asarray(clean)
    return res


def clr(X):
    to_return = np.zeros(X.shape)
    m = X.shape[0]
    for i in range(0,m):
        x_gmean = stats.gmean(X[:,i])
        to_return[:,i] = np.log(X[:,i] / x_gmean)

    return to_return
