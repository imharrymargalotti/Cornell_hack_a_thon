import Tim
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
    clf = IsolationForest(behaviour='new', max_samples=100, contamination='auto')
    clf.fit(H)
    for x in range(3):  # loops case
        case = AllCases[x]
        h, d = ProcessData(case, y[x])
        plotDist(h, d, clf, (str(x) + "_plot"))

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
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def geo_mean(iterable):
    #nz = np.nonzero(iterable)
    #print(nz)
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


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
    c = np.asmatrix(col)
    data = np.asarray(rows)
    d = np.asmatrix(data)

    return c, d


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
    # print(X[:,1])
    m = X.shape[1]
    u = m-1
    for i in range(0,m):
        # print(i)
        col = X[:,i]
        x_gmean = geo_mean(X[:,i])
        print(x_gmean)
        to_return[:,i] = np.log(X[:,i].reshape(X.shape[0]) / x_gmean)
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
    c_path = os.environ.get("CCSI")
    s_path = os.environ.get("SAMEA")
    e_path = os.environ.get("ERR")

    e_id, e_res = read_in_data(e_path)

    # print(e_res.shape)
    # print(e_res)
    #print(e_res.shape)
    # col = e_res[:,1]
    # print(col.shape)
    # nz = np.nonzero(col)
    # #print(nz)
    # use = col[nz]
    # print(use)
    # x =geo_mean(use)
    # print(x)


    # clr_e = clr(e_res)
    # print(clr_e)


    print(e_res.shape)
    s_id, s_res = read_in_data(s_path)
    clr_s = clr(s_res)
    print(s_res.shape)
    c_id, c_res = read_in_data(c_path)
    clr_c = clr(c_res)
    print(c_res.shape)

    all_data = []
    all_data.append(e_res.T)
    all_data.append(c_res.T)
    all_data.append(s_res.T)

    CCSI = "CCIS00146684ST-4-0  CCIS00281083ST-3-0  CCIS02124300ST-4-0  CCIS02379307ST-4-0  CCIS02856720ST-4-0  CCIS03473770ST-4-0  CCIS03857607ST-4-0  CCIS05314658ST-4-0  CCIS06260551ST-3-0  CCIS07277498ST-4-0  CCIS07539127ST-4-0  CCIS07648107ST-4-0  CCIS09568613ST-4-0  CCIS10706551ST-3-0  CCIS11015875ST-4-0  CCIS11354283ST-4-0  CCIS11362406ST-4-0  CCIS11558985ST-4-0  CCIS12370844ST-4-0  CCIS12656533ST-4-0  CCIS13047523ST-4-0  CCIS14449628ST-4-0  CCIS15704761ST-4-0  CCIS15794887ST-4-0  CCIS16326685ST-4-0  CCIS16383318ST-4-0  CCIS16561622ST-4-0  CCIS17669415ST-4-0  CCIS20795251ST-4-0  CCIS21126322ST-4-0  CCIS21278152ST-4-0  CCIS22416007ST-4-0  CCIS22958137ST-20-0  CCIS23164343ST-4-0  CCIS24254057ST-4-0  CCIS27304052ST-3-0  CCIS27927933ST-4-0  CCIS29210128ST-4-0  CCIS29688262ST-20-0  CCIS32452666ST-4-0  CCIS33816588ST-4-0  CCIS34055159ST-4-0  CCIS34604008ST-4-0  CCIS35100175ST-4-0  CCIS36699628ST-4-0  CCIS36797902ST-4-0  CCIS38765456ST-20-0  CCIS40244499ST-3-0  CCIS41222843ST-4-0  CCIS41288781ST-4-0  CCIS41548810ST-4-0  CCIS41692898ST-4-0  CCIS44093303ST-4-0  CCIS44757994ST-4-0  CCIS45571137ST-3-0  CCIS45793747ST-4-0  CCIS46047672ST-4-0  CCIS46467422ST-4-0  CCIS47284573ST-4-0  CCIS48174381ST-4-0  CCIS48725289ST-4-0  CCIS50003399ST-4-0  CCIS50148151ST-4-0  CCIS50471204ST-4-0  CCIS51595129ST-4-0  CCIS52370277ST-4-0  CCIS53043478ST-4-0  CCIS53355328ST-4-0  CCIS55230578ST-4-0  CCIS58234805ST-4-0  CCIS59132091ST-4-0  CCIS61287323ST-4-0  CCIS62605362ST-3-0  CCIS62794166ST-4-0  CCIS63448910ST-4-0  CCIS63468405ST-4-0  CCIS63910149ST-4-0  CCIS64773582ST-4-0  CCIS64785924ST-20-0  CCIS65479369ST-4-0  CCIS71301801ST-4-0  CCIS71578391ST-4-0  CCIS72607085ST-4-0  CCIS74239020ST-4-0  CCIS76845094ST-20-0  CCIS77252613ST-4-0  CCIS78100604ST-4-0  CCIS78318719ST-4-0  CCIS79210440ST-3-0  CCIS80834637ST-4-0  CCIS81139242ST-4-0  CCIS81887263ST-4-0  CCIS82146115ST-4-0  CCIS82507866ST-3-0  CCIS82944710ST-20-0  CCIS83870198ST-4-0  CCIS84543192ST-4-0  CCIS85214191ST-3-0  CCIS87116798ST-4-0  CCIS87167916ST-4-0  CCIS87252800ST-4-0  CCIS87605453ST-4-0  CCIS88007743ST-4-0  CCIS88317640ST-4-0  CCIS90164298ST-4-0  CCIS91228662ST-4-0  CCIS93040568ST-20-0  CCIS94417875ST-3-0  CCIS94603952ST-4-0  CCIS95097901ST-4-0  CCIS95409808ST-4-0  CCIS98482370ST-3-0  CCIS98512455ST-4-0  CCIS98832363ST-4-0"
    clean_CCSI = CCSI.split("  ")
    SAMEA = "SAMEA3136623    SAMEA3136624    SAMEA3136625    SAMEA3136626    SAMEA3136627    SAMEA3136628    SAMEA3136629    SAMEA3136630    SAMEA3136631    SAMEA3136632    SAMEA3136633    SAMEA3136634    SAMEA3136635    SAMEA3136636    SAMEA3136637    SAMEA3136638    SAMEA3136639    SAMEA3136640    SAMEA3136641    SAMEA3136642    SAMEA3136643    SAMEA3136644    SAMEA3136645    SAMEA3136646    SAMEA3136647    SAMEA3136648    SAMEA3136649    SAMEA3136650    SAMEA3136651    SAMEA3136652    SAMEA3136653    SAMEA3136654    SAMEA3136655    SAMEA3136656    SAMEA3136657    SAMEA3136658    SAMEA3136659    SAMEA3136660    SAMEA3136661    SAMEA3136662    SAMEA3136663    SAMEA3136664    SAMEA3136665    SAMEA3136666    SAMEA3136667    SAMEA3136668    SAMEA3136669    SAMEA3136670    SAMEA3136671    SAMEA3136672    SAMEA3136673    SAMEA3136674    SAMEA3136675    SAMEA3136676    SAMEA3136677    SAMEA3136678    SAMEA3136679    SAMEA3136724    SAMEA3136725    SAMEA3136726    SAMEA3136727    SAMEA3136728    SAMEA3136729    SAMEA3136730    SAMEA3136731    SAMEA3136732    SAMEA3136733    SAMEA3136734    SAMEA3136735    SAMEA3136736    SAMEA3136737    SAMEA3136738    SAMEA3136739    SAMEA3136740    SAMEA3136741    SAMEA3136742    SAMEA3136743    SAMEA3136744    SAMEA3136745    SAMEA3136746    SAMEA3136747    SAMEA3136748    SAMEA3136749    SAMEA3136750    SAMEA3136751    SAMEA3136752    SAMEA3136753    SAMEA3136754    SAMEA3136755    SAMEA3136756    SAMEA3136757    SAMEA3136758    SAMEA3136759    SAMEA3136760    SAMEA3136761    SAMEA3136762    SAMEA3136763    SAMEA3136764    SAMEA3136765    SAMEA3136766    SAMEA3136767    SAMEA3136768    SAMEA3136769    SAMEA3178936    SAMEA3178937    SAMEA3178938    SAMEA3178939    SAMEA3178940    SAMEA3178943"
    clean_SAMEA = SAMEA.split("    ")

    e_y = read_in_y()
    # print(e_y)
    # print(len(e_y))
    cy_path = os.environ.get("CY")
    c_y = cc_y(clean_CCSI, cy_path)
    # print(c_y)
    # print(len(c_y))
    sam_path = os.environ.get("SY")
    s_y = sam_y(clean_SAMEA, sam_path)
    # print(s_y)
    # print(len(s_y))

    #------------------------------------------------PIPELINE-------------------------------------------------------------------------------------------------------------
    AnomolyDetector(all_data, s_id)
    

main()