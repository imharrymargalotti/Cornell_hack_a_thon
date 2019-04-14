import numpy as np
import os
import sklearn
from scipy import stats
from Deploy import environset
import pandas as pd
import numpy as np
import phate
import scprep
from sklearn.ensemble import IsolationForest
import seaborn
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import roc_curve, auc


environset.set_paths()
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

    return col, data.T

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

def ProcessData(data,y_train):
    hArray=[]
    dArray=[]
    shape,garbage=data.shape
    for x in range(shape):
        if(y_train[x]==0):
            hArray.append(data[x])
        else:
            dArray.append(data[x])
    hArray=np.array(hArray)
    dArray=np.array(dArray)


    return hArray,dArray
def plotDist(hArray,dArray,IF,name):
    dArrayScore=IF.score_samples(dArray)
    hArrayScore=IF.score_samples(hArray)
    seaborn.distplot(dArrayScore, color="skyblue", label="Disease")
    seaborn.distplot(hArrayScore, color="red", label="Healthy")
    plt.savefig(name,transparant=True)

def plotRoc(fpr,tpr,name):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw='lw', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(name, transparant=True)
    plt.show()


col,data=read_in_ERR()
y_train=read_in_y()
hArray,dArray=ProcessData(data,y_train)
clf = IsolationForest(behaviour='new', max_samples=100, contamination='auto')
clf.fit(hArray)
plotDist(hArray,dArray,clf,"/Users/jeffpage/Desktop/CowPY/images/image1.png")








