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


def read_in_data(path):
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
    ones = np.ones(X.shape)
    m = X.shape[1]
    use_me = X+ones
    for i in range(0,m):
        x_gmean = stats.gmean(use_me[:,i])
        print(x_gmean)
        to_return[:,i] = np.log(use_me[:,i] / x_gmean)
    return to_return



def cc_y(cc, path):
    with open(path) as f:
        lines = f.readlines()
    f.close()
    c_cc = [x.strip() for x in lines]

    full_cc = []
    for c in c_cc:
        full_cc.append(c.split(","))
    del full_cc[0]

    cc_done = []

    for c in range(0,len(cc)):
        for row in full_cc:
            if row[1] == cc[c]:
                #print(row[1])
                cc_done.append(row)
                #del cc[c]
                break

    #print(cc_done)
    cc_y = []
    for c in cc_done:
        if c[8] == "Normal":
            cc_y.append(0)
        else:
            cc_y.append(1)
    return cc_y


    # with open("/Users/timc/PycharmProjects/CowPY/Data/CC.csv", "w") as f:
    #     for row in cc_done:
    #         string = ""
    #         for item in range(len(row)):
    #             if item==len(row)-1:
    #                 string+=str(row[item])
    #             elif item!=0:
    #                 string += str(row[item])+","
    #         string+="\n"
    #         f.write(string)
    # f.close()


def sam_y(sa, path):
    with open(path) as f:
        slines= f.readlines()
    f.close()

    c_sa = [x.strip() for x in slines]

    full_cc = []
    for s in c_sa:
        full_cc.append(s.split(","))
    del full_cc[0]



    sa_done = []
    for s in range(0, len(sa)):
        for row in full_cc:
            if row[2] == sa[s]:
                sa_done.append(row)
                break


    sa_y = []
    for s in sa_done:
        if s[20] == "Stool sample from controls":
            sa_y.append(0)
        else:
            sa_y.append(1)
    return sa_y

    # with open("/Users/timc/PycharmProjects/CowPY/Data/SA.csv", "w") as f:
    #     for row in sa_done:
    #         string = ""
    #         for item in range(len(row)):
    #             if item==len(row)-1:
    #                 string+=str(row[item])
    #             elif item!=0:
    #                 string += str(row[item])+","
    #         string+="\n"
    #         f.write(string)
    # f.close()


def main():

    print()


main()