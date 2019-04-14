#import Tim
from Deploy import environset
import os
import numpy as np
from scipy import stats
import numpy as np
import os
import sklearn
from scipy import stats
from Deploy import environset
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn
import matplotlib.pyplot as plt
from random import randint



environset.set_paths()


def AnomolyDetector(AllCases, y):
    # xtrain is yu healthy
    # assumes largest set 'to train on' is first
    H, D = ProcessData(AllCases[0], y[0])
    # print(H.shape)
    test = H.reshape((53,849))
    # print(test.shape)
    # print(test)
    clf = IsolationForest(max_samples='auto')
    clf.fit(test)
    name = ""
    for x in range(3):  # loops case
        case = AllCases[x]
        h, d = ProcessData(case, y[x])
        if (x==0):
            name = "e_data"
        elif(x==1):
            name = "c_data"
        else:
            name = "s_data"

        plotDist(h, d, clf, (str(x) + name))

def ProcessData(data,y_train):
    hArray=[]
    dArray=[]
    shape,garbage=data.shape
    for x in range(shape):
        if(y_train[x]==0):
            hArray.append(data[x,:])
        else:
            dArray.append(data[x,:])
    #print(hArray)
    hArray=np.array(hArray)
    dArray=np.array(dArray)


    return hArray,dArray


def plotDist(hArray,dArray,IF,name):
    # print("in dist")

    dArrayScore=IF.score_samples(dArray)
    hArrayScore=IF.score_samples(hArray)
    seaborn.distplot(dArrayScore, color="skyblue", label="Disease")
    seaborn.distplot(hArrayScore, color="red", label="Healthy")
    plt.savefig(name,transparant=True)
    plt.show()

def clr(X):
    to_return = np.zeros(X.shape)
    ones = np.ones(X.shape)
    m = X.shape[0]
    use_me = X+ones
    for i in range(0,m):
        x_gmean = stats.gmean(use_me[i,:])
        to_return[i,:] = np.log(use_me[i,:] / x_gmean)
    return to_return



# def plotRoc(fpr,tpr,name):
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()




def main():
    data = np.load('info.npz')
    # print(data['e_data'].shape)
    # sdata, edata, cdata
    #s_train
    AllData = []
    AllData.append(clr(data["e_data"]))
    AllData.append(clr(data["c_data"]))
    AllData.append(clr(data["s_data"]))

    AllY = []
    data2 = np.load('y_trains.npz')
    AllY.append(data2["e_train"])
    AllY.append(data2["c_train"])
    AllY.append(data2["s_train"])

    AnomolyDetector(AllData, AllY)

main()
