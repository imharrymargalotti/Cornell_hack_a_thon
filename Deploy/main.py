#import Tim
import copy

from sklearn.metrics import auc, roc_auc_score

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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
    clf = IsolationForest(n_estimators=500, max_samples='auto')
    clf.fit(test)
    name = ""
    for x in range(3):  # loops case
        case = AllCases[x]
        h, d = ProcessData(case, y[x])
        # print(y[x].shape)
        if (x==0):
            name = "China_anomolyTrainedOnChinaHealthy"
        elif(x==1):
            name = "Germany_anomolyTrainedOnChinaHealthy"
        else:
            name = "US_anomolyTrainedOnChinaHealthy"

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


def plotRoc(fpr,tpr,name):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(name, transparant=True)
    plt.show()


def RandForest(AllData, y):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    X = AllData[0]
    y_train = y[0]
    clf.fit(X, y_train)  # TRAIN ON YU
    name = ""
    for x in range(3):  # loops case
        if (x == 0):
            name = "China_dataTrainedOnChina"
        elif (x == 1):
            name = "Germany_dataTrainedOnChina"
        else:
            name = "US_dataTrainedOnChina"
        case = AllData[x]
        y_predict = clf.predict_proba(case)[:,1]


        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[x], y_predict)
        plotRoc(fpr, tpr, str(x+10)+name+".png")

def PooledRF(AllData, y):
    x = AllData[0]
    x2 = AllData[2]
    y1 = y[0]
    y2 = y[2]
    rows,cols = x.shape
    row2,col2 = x2.shape
    crow = rows+row2

    bigData=np.zeros((crow, 849))

    for i in range(rows):
        bigData[i] = x[i]

    for i in range(rows, crow):
        bigData[i] = x2[i-rows]

    y_total = []
    for num in y1:
        y_total.append(num)

    for num in y2:
        y_total.append(num)

    y_fin = np.asarray(y_total)
    # print(y_fin)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(bigData,y_fin)
    name = ""
    for x in range(3):  # loops case
        if (x == 0):
            name = "China_data_train_US_data_train_ChinaAndUSy"
        elif (x == 1):
            name = "Germany_data_train_ChinaAndUS"
        else:
            name = "US_data_train_ChinaAndUS"
        case = AllData[x]
        y_predict = clf.predict_proba(case)[:,1]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[x], y_predict)
        plotRoc(fpr, tpr, str(x + 10) + name + ".png")



def featExtraction(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(X_train,y_train)
    RocValues = []
    tempX_train = X_train
    rows,cols = X_train.shape

    for J in range(cols):
        X_train = copy.deepcopy(tempX_train) #reset
        np.random.shuffle(X_train[:,J]) #shuffle j'th col
        y_predict = clf.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, y_predict)
        roc_auc = auc(fpr, tpr)
        RocValues.append(roc_auc)
    return RocValues

def bubbleSort(alist, headers):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                # numbers
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
                #end numbers
                temph = headers[i]
                headers[i] = headers[i + 1]
                headers[i + 1] = temph


def readHeaders():
    header = "Streptococcus anginosus [ref_mOTU_v2_0004],Enterobacteriaceae sp. [ref_mOTU_v2_0036],Citrobacter sp. [ref_mOTU_v2_0076],Klebsiella michiganensis/oxytoca [ref_mOTU_v2_0079],Enterococcus faecalis [ref_mOTU_v2_0116],Lactobacillus salivarius [ref_mOTU_v2_0125],Dielma fastidiosa [ref_mOTU_v2_0138],Streptococcus constellatus/intermedius [ref_mOTU_v2_0143],Streptococcus parasanguinis [ref_mOTU_v2_0144],Streptococcus sp. HSISM1 [ref_mOTU_v2_0145],Peptostreptococcus anaerobius [ref_mOTU_v2_0148],Bifidobacterium longum [ref_mOTU_v2_0150],Bifidobacterium breve [ref_mOTU_v2_0157],Klebsiella sp. [ref_mOTU_v2_0160],Lactobacillus ruminis [ref_mOTU_v2_0167],Lactococcus lactis [ref_mOTU_v2_0182],Streptococcus vestibularis [ref_mOTU_v2_0198],Streptococcus salivarius [ref_mOTU_v2_0199],Streptococcus thermophilus [ref_mOTU_v2_0219],Lactobacillus casei/paracasei [ref_mOTU_v2_0226],Megasphaera elsdenii [ref_mOTU_v2_0252],Streptococcus sp. [ref_mOTU_v2_0261],Enterobacter sp. [ref_mOTU_v2_0265],Bacteroides stercoris [ref_mOTU_v2_0275],Prevotella nigrescens [ref_mOTU_v2_0276],Streptococcus sanguinis [ref_mOTU_v2_0279],Ruminococcus gnavus [ref_mOTU_v2_0280],Ruminococcus lactaris [ref_mOTU_v2_0281],Bacteroides fragilis [ref_mOTU_v2_0286],Bacteroides fragilis [ref_mOTU_v2_0287],Streptococcus mutans [ref_mOTU_v2_0289],Bacteroides nordii [ref_mOTU_v2_0302],Coprococcus sp. [ref_mOTU_v2_0303],Streptococcus anginosus [ref_mOTU_v2_0351],Streptococcus oralis [ref_mOTU_v2_0356],Haemophilus parainfluenzae [ref_mOTU_v2_0358],Lactococcus lactis [ref_mOTU_v2_0371],Enterococcus faecium [ref_mOTU_v2_0372],Acidaminococcus intestini [ref_mOTU_v2_0391],Streptococcus sp. [ref_mOTU_v2_0416],Anaerococcus obesiensis/vaginalis [ref_mOTU_v2_0429],Bacteroides barnesiae [ref_mOTU_v2_0452],Bacteroides massiliensis [ref_mOTU_v2_0455],Bacteroides salyersiae [ref_mOTU_v2_0458],Blautia wexlerae [ref_mOTU_v2_0466],Clostridium saccharogumia [ref_mOTU_v2_0473],Megamonas funiformis/rupellensis [ref_mOTU_v2_0502],Prevotella intermedia [ref_mOTU_v2_0515],Prevotella oris [ref_mOTU_v2_0520],Solobacterium moorei [ref_mOTU_v2_0531],Veillonella atypica [ref_mOTU_v2_0561],Lactobacillus mucosae [ref_mOTU_v2_0568],Enterococcus durans [ref_mOTU_v2_0598],Bifidobacterium dentium [ref_mOTU_v2_0631],Bifidobacterium pseudocatenulatum [ref_mOTU_v2_0632],Bifidobacterium catenulatum/kashiwanohense [ref_mOTU_v2_0633],Eggerthella lenta [ref_mOTU_v2_0642],Clostridium innocuum [ref_mOTU_v2_0643],Enterococcus faecium/hirae [ref_mOTU_v2_0654],Streptococcus salivarius [ref_mOTU_v2_0656],Streptococcus anginosus [ref_mOTU_v2_0687],Bacteroides cellulosilyticus/timonensis [ref_mOTU_v2_0692],Lactobacillus gasseri [ref_mOTU_v2_0725],Atopobium parvulum [ref_mOTU_v2_0741],Fusobacterium nucleatum s. vincentii [ref_mOTU_v2_0754],Sutterella wadsworthensis [ref_mOTU_v2_0767],Alistipes onderdonkii [ref_mOTU_v2_0775],Fusobacterium nucleatum s. animalis [ref_mOTU_v2_0776],Fusobacterium nucleatum s. nucleatum [ref_mOTU_v2_0777],Bifidobacterium bifidum [ref_mOTU_v2_0786],Streptococcus equinus [ref_mOTU_v2_0793],Oscillibacter sp. KLE 1728 [ref_mOTU_v2_0858],Blautia sp. KLE 1732 [ref_mOTU_v2_0859],Clostridium sp. KLE 1755 [ref_mOTU_v2_0860],Hungatella hathewayi [ref_mOTU_v2_0882],Clostridium scindens [ref_mOTU_v2_0883],Anaerotruncus colihominis [ref_mOTU_v2_0884],Erysipelotrichaceae sp. [ref_mOTU_v2_0885],Clostridium boltae/clostridioforme [ref_mOTU_v2_0886],Lachnospiraceae bacterium 3_1_57FAA_CT1 [ref_mOTU_v2_0887],Bacteroides thetaiotaomicron [ref_mOTU_v2_0897],Bacteroides dorei/vulgatus [ref_mOTU_v2_0898],Bacteroides rodentium/uniformis [ref_mOTU_v2_0899],Parabacteroides goldsteinii [ref_mOTU_v2_0901],Phascolarctobacterium succinatutens [ref_mOTU_v2_0947],Megasphaera sp. [ref_mOTU_v2_0965],Dorea formicigenerans [ref_mOTU_v2_0973],Clostridium butyricum [ref_mOTU_v2_0978],Clostridium clostridioforme [ref_mOTU_v2_0979],Clostridium clostridioforme [ref_mOTU_v2_0980],Veillonella parvula [ref_mOTU_v2_1042],Lactobacillus fermentum [ref_mOTU_v2_1050],Bacteroides xylanisolvens [ref_mOTU_v2_1072],Bacteroides fragilis/ovatus [ref_mOTU_v2_1073],Bacteroidales sp. [ref_mOTU_v2_1074],Lactobacillus reuteri [ref_mOTU_v2_1076],Porphyromonadaceae sp. [ref_mOTU_v2_1091],Clostridium perfringens [ref_mOTU_v2_1117],Mogibacterium timidum [ref_mOTU_v2_1136],Parvimonas micra [ref_mOTU_v2_1145],Bilophila wadsworthia [ref_mOTU_v2_1149],Bifidobacterium adolescentis [ref_mOTU_v2_1156],Bifidobacterium ruminantium [ref_mOTU_v2_1159],Bifidobacterium sp. [ref_mOTU_v2_1180],Hafnia alvei [ref_mOTU_v2_1202],Bifidobacterium angulatum [ref_mOTU_v2_1285],Akkermansia muciniphila [ref_mOTU_v2_1301],Anaerostipes hadrus [ref_mOTU_v2_1309],Ruminococcus torques [ref_mOTU_v2_1376],Flavonifractor plautii [ref_mOTU_v2_1377],Parabacteroides merdae [ref_mOTU_v2_1378],Faecalibacterium prausnitzii [ref_mOTU_v2_1379],Clostridium saccharolyticum [ref_mOTU_v2_1380],Anaerostipes caccae [ref_mOTU_v2_1381],Bacteroides caccae [ref_mOTU_v2_1382],Collinsella aerofaciens [ref_mOTU_v2_1383],Methanobrevibacter smithii [ref_mOTU_v2_1384],Methanobrevibacter smithii [ref_mOTU_v2_1385],Eubacterium siraeum [ref_mOTU_v2_1387],Desulfovibrio sp. [ref_mOTU_v2_1394],Eubacterium sp. [ref_mOTU_v2_1395],Fusobacterium ulcerans [ref_mOTU_v2_1396],Streptococcus sp. 2_1_36FAA [ref_mOTU_v2_1399],Coprobacillus sp. [ref_mOTU_v2_1401],Fusobacterium sp. oral taxon 370 [ref_mOTU_v2_1403],Fusobacterium gonidiaformans [ref_mOTU_v2_1404],Parasutterella excrementihominis [ref_mOTU_v2_1405],Bacteroides finegoldii [ref_mOTU_v2_1409],Bacteroides eggerthii [ref_mOTU_v2_1410],Eubacterium rectale [ref_mOTU_v2_1416],Roseburia intestinalis [ref_mOTU_v2_1427],Blautia hansenii [ref_mOTU_v2_1428],Faecalitalea cylindroides [ref_mOTU_v2_1470],Clostridium symbiosum [ref_mOTU_v2_1475],Alistipes finegoldii [ref_mOTU_v2_1481],Turicibacter sanguinis [ref_mOTU_v2_1493],Porphyromonas asaccharolytica [ref_mOTU_v2_1517],Streptococcus australis [ref_mOTU_v2_1524],Prevotella stercorea [ref_mOTU_v2_1551],Alistipes timonensis [ref_mOTU_v2_1593],Alistipes senegalensis [ref_mOTU_v2_1594],Senegalimassilia anaerobia [ref_mOTU_v2_1606],Bacteroides faecis [ref_mOTU_v2_1706],Enorma massiliensis [ref_mOTU_v2_1824],Alistipes obesi [ref_mOTU_v2_1825],Blautia producta [ref_mOTU_v2_1889],Parabacteroides gordonii [ref_mOTU_v2_2090],Porphyromonas somerae [ref_mOTU_v2_2101],Porphyromonas uenonis [ref_mOTU_v2_2102],Faecalicoccus pleomorphus [ref_mOTU_v2_2178],Ruminococcus bicirculans [ref_mOTU_v2_2358],Butyricicoccus pullicaecorum [ref_mOTU_v2_2510],cand. Alistipes marseilloanorexicus [ref_mOTU_v2_2553],Holdemania massiliensis [ref_mOTU_v2_2557],Megasphaera massiliensis [ref_mOTU_v2_2684],Clostridiales bacterium VE202-09 [ref_mOTU_v2_2688],Clostridiales bacterium VE202-14 [ref_mOTU_v2_2689],Bacteroides stercorirosoris [ref_mOTU_v2_2726],Eubacterium ramulus [ref_mOTU_v2_2795],Phascolarctobacterium sp. [ref_mOTU_v2_2805],Phascolarctobacterium sp. [ref_mOTU_v2_2806],Clostridium paraputrificum [ref_mOTU_v2_2890],Dorea longicatena [ref_mOTU_v2_2893],Intestinimonas butyriciproducens [ref_mOTU_v2_2968],Adlercreutzia equolifaciens [ref_mOTU_v2_3198],Mitsuokella jalaludinii [ref_mOTU_v2_3339],Clostridium sp. JCC [ref_mOTU_v2_3353],Alistipes inops [ref_mOTU_v2_3597],bacterium OL-1 [ref_mOTU_v2_3607],bacterium LF-3 [ref_mOTU_v2_3608],Oscillibacter sp. ER4 [ref_mOTU_v2_3624],Ruminococcus champanellensis [ref_mOTU_v2_3773],butyrate-producing bacterium SS3/4 [ref_mOTU_v2_3825],Blautia producta [ref_mOTU_v2_4020],Pyramidobacter piscolens [ref_mOTU_v2_4064],Blautia obeum [ref_mOTU_v2_4202],Dorea longicatena [ref_mOTU_v2_4203],Eubacterium ventriosum [ref_mOTU_v2_4204],Desulfovibrio piger [ref_mOTU_v2_4205],Eubacterium hallii [ref_mOTU_v2_4207],Ruminococcus callidus [ref_mOTU_v2_4209],Coprococcus eutactus [ref_mOTU_v2_4210],Faecalibacterium prausnitzii [ref_mOTU_v2_4211],Clostridium sp. L2-50 [ref_mOTU_v2_4212],Clostridium leptum [ref_mOTU_v2_4234],Clostridium spiroforme [ref_mOTU_v2_4235],Alistipes putredinis [ref_mOTU_v2_4266],Intestinibacter bartlettii [ref_mOTU_v2_4268],Synergistes sp. 3_1_syn1 [ref_mOTU_v2_4289],Fusobacterium mortiferum [ref_mOTU_v2_4310],Fusobacterium varium [ref_mOTU_v2_4311],Bacteroides coprocola [ref_mOTU_v2_4312],Coprococcus comes [ref_mOTU_v2_4313],Bacteroides intestinalis [ref_mOTU_v2_4321],Blautia hydrogenotrophica [ref_mOTU_v2_4324],Bacteroides pectinophilus [ref_mOTU_v2_4341],Bacteroides plebeius [ref_mOTU_v2_4343],Tyzzerella nexilis [ref_mOTU_v2_4366],Mitsuokella multacida [ref_mOTU_v2_4368],Butyrivibrio crossotus [ref_mOTU_v2_4383],Eubacterium eligens [ref_mOTU_v2_4389],Clostridium asparagiforme [ref_mOTU_v2_4394],Holdemanella biformis [ref_mOTU_v2_4395],Collinsella intestinalis [ref_mOTU_v2_4400],Prevotella copri [ref_mOTU_v2_4448],Holdemania filiformis [ref_mOTU_v2_4459],Veillonella dispar [ref_mOTU_v2_4469],Bacteroides coprophilus [ref_mOTU_v2_4472],Ruminococcaceae bacterium D16 [ref_mOTU_v2_4480],Gemella morbillorum [ref_mOTU_v2_4513],Roseburia hominis [ref_mOTU_v2_4572],Acidaminococcus fermentans [ref_mOTU_v2_4591],Dialister invisus [ref_mOTU_v2_4598],Peptostreptococcus stomatis [ref_mOTU_v2_4614],Porphyromonas uenonis [ref_mOTU_v2_4616],Roseburia inulinivorans [ref_mOTU_v2_4632],Alloprevotella tannerae [ref_mOTU_v2_4636],Granulicatella adiacens [ref_mOTU_v2_4659],Ruminococcus torques [ref_mOTU_v2_4718],Blautia obeum [ref_mOTU_v2_4719],Ruminococcus bromii [ref_mOTU_v2_4720],Lachnospiraceae bacterium 1_4_56FAA [ref_mOTU_v2_4723],Erysipelotrichaceae sp. [ref_mOTU_v2_4724],Subdoligranulum sp. 4_3_54A2FAA [ref_mOTU_v2_4738],Megasphaera micronuciformis [ref_mOTU_v2_4840],Odoribacter splanchnicus [ref_mOTU_v2_4846],Alistipes shahii [ref_mOTU_v2_4873],Coprococcus catus [ref_mOTU_v2_4874],Faecalibacterium prausnitzii [ref_mOTU_v2_4875],Alistipes indistinctus [ref_mOTU_v2_4879],Barnesiella intestinihominis [ref_mOTU_v2_4880],Clostridium citroniae [ref_mOTU_v2_4882],Collinsella tanakaei [ref_mOTU_v2_4884],Dialister succinatiphilus [ref_mOTU_v2_4885],Slackia piriformis [ref_mOTU_v2_4888],Sutterella wadsworthensis [ref_mOTU_v2_4889],Faecalibacterium prausnitzii [ref_mOTU_v2_4910],Paraprevotella clara [ref_mOTU_v2_4942],Paraprevotella xylaniphila [ref_mOTU_v2_4943],Succinatimonas hippei [ref_mOTU_v2_4944],Bacteroides clarus [ref_mOTU_v2_4945],Bacteroides fluxus [ref_mOTU_v2_4946],Parvimonas sp. [ref_mOTU_v2_4961],Veillonella sp. [ref_mOTU_v2_5068],Parvimonas sp. [ref_mOTU_v2_5245],Parabacteroides johnsonii [ref_mOTU_v2_5296],unknown Clostridia [meta_mOTU_v2_5309],unknown Bacteroidales [meta_mOTU_v2_5329],unknown Ruminococcaceae [meta_mOTU_v2_5330],unknown Clostridium [meta_mOTU_v2_5331],unknown Clostridium [meta_mOTU_v2_5336],unknown Dialister [meta_mOTU_v2_5337],unknown Clostridiales [meta_mOTU_v2_5339],unknown Clostridiales [meta_mOTU_v2_5341],unknown Clostridium [meta_mOTU_v2_5343],unknown Clostridia [meta_mOTU_v2_5344],unknown Clostridiales [meta_mOTU_v2_5347],Oscillibacter sp. 57_20 [meta_mOTU_v2_5351],unknown Clostridium [meta_mOTU_v2_5353],unknown Roseburia [meta_mOTU_v2_5354],unknown Anaeromassilibacillus [meta_mOTU_v2_5357],unknown Clostridiales [meta_mOTU_v2_5362],Clostridium sp. CAG:127 [meta_mOTU_v2_5364],unknown Clostridium [meta_mOTU_v2_5366],unknown Eggerthellaceae [meta_mOTU_v2_5373],unknown Bacteroidales [meta_mOTU_v2_5375],unknown Coprococcus [meta_mOTU_v2_5379],unknown Clostridiaceae [meta_mOTU_v2_5382],Azospirillum sp. CAG:239 [meta_mOTU_v2_5386],Clostridium sp. CAG:440 [meta_mOTU_v2_5389],unknown Ruminococcus [meta_mOTU_v2_5392],unknown Sutterellaceae [meta_mOTU_v2_5393],unknown Clostridiales [meta_mOTU_v2_5396],unknown Clostridiales [meta_mOTU_v2_5397],Prevotella sp. CAG:279 [meta_mOTU_v2_5405],unknown Clostridiales [meta_mOTU_v2_5408],unknown Clostridiales [meta_mOTU_v2_5411],unknown Clostridiales [meta_mOTU_v2_5412],unknown Clostridium [meta_mOTU_v2_5413],unknown Clostridiales [meta_mOTU_v2_5415],unknown Alistipes [meta_mOTU_v2_5423],unknown Porphyromonas [meta_mOTU_v2_5431],unknown Dehalococcoidales [meta_mOTU_v2_5435],unknown Butyricicoccus [meta_mOTU_v2_5437],unknown Clostridiales [meta_mOTU_v2_5442],Ruminococcus sp. CAG:177 [meta_mOTU_v2_5453],unknown Eubacterium [meta_mOTU_v2_5463],Clostridium sp. CAG:793 [meta_mOTU_v2_5464],unknown Clostridiales [meta_mOTU_v2_5466],uncultured Eubacterium sp. [meta_mOTU_v2_5477],unknown Dehalococcoidales [meta_mOTU_v2_5481],Clostridiales bacterium 41_21_two_genomes [meta_mOTU_v2_5482],Clostridiales bacterium S5-A14a [meta_mOTU_v2_5486],Clostridium sp. CAG:571 [meta_mOTU_v2_5498],Clostridium sp. CAG:594 [meta_mOTU_v2_5501],unknown Prevotella [meta_mOTU_v2_5502],unknown Clostridium [meta_mOTU_v2_5505],unknown Clostridiales [meta_mOTU_v2_5511],unknown Clostridiales [meta_mOTU_v2_5513],unknown Clostridiales [meta_mOTU_v2_5514],unknown Clostridiales [meta_mOTU_v2_5517],unknown Bacteroidales [meta_mOTU_v2_5520],unknown Firmicutes [meta_mOTU_v2_5525],unknown Synergistaceae [meta_mOTU_v2_5538],unknown Clostridiales [meta_mOTU_v2_5540],Libanicoccus massiliensis [meta_mOTU_v2_5543],unknown Phascolarctobacterium [meta_mOTU_v2_5554],unknown Prevotella [meta_mOTU_v2_5555],unknown Clostridium [meta_mOTU_v2_5561],unknown Prevotellaceae [meta_mOTU_v2_5568],unknown Clostridiales [meta_mOTU_v2_5569],unknown Clostridiales [meta_mOTU_v2_5571],unknown Clostridiales [meta_mOTU_v2_5583],unknown Bacteroidales [meta_mOTU_v2_5584],unknown Clostridium [meta_mOTU_v2_5585],unknown Clostridiales [meta_mOTU_v2_5591],unknown Clostridiales [meta_mOTU_v2_5603],unknown Bacteroides [meta_mOTU_v2_5618],unknown Ruminococcus [meta_mOTU_v2_5630],Mailhella massiliensis [meta_mOTU_v2_5632],unknown Clostridiales [meta_mOTU_v2_5641],unknown Streptococcus [meta_mOTU_v2_5642],unknown Bacteroidales [meta_mOTU_v2_5647],unknown Bacteria [meta_mOTU_v2_5651],Azospirillum sp. 51_20 [meta_mOTU_v2_5652],unknown Clostridiales [meta_mOTU_v2_5653],unknown Bacteroidales [meta_mOTU_v2_5655],unknown Anaeromassilibacillus [meta_mOTU_v2_5656],unknown Firmicutes [meta_mOTU_v2_5660],unknown Clostridiales [meta_mOTU_v2_5661],unknown Prevotella [meta_mOTU_v2_5668],unknown Clostridiales [meta_mOTU_v2_5669],unknown Odoribacter [meta_mOTU_v2_5670],unknown Sutterella [meta_mOTU_v2_5677],Clostridium sp. CAG:273 [meta_mOTU_v2_5681],Mycoplasma sp. CAG:611 [meta_mOTU_v2_5692],Clostridium sp. CAG:306 [meta_mOTU_v2_5698],Butyrivibrio sp. CAG:318 [meta_mOTU_v2_5702],Bacteroides sp. CAG:462 [meta_mOTU_v2_5709],unknown Clostridiales [meta_mOTU_v2_5710],Prevotella sp. CAG:891 [meta_mOTU_v2_5711],unknown Clostridiales [meta_mOTU_v2_5712],Clostridium sp. CAG:590 [meta_mOTU_v2_5713],Clostridium sp. 27_14 [meta_mOTU_v2_5718],unknown Firmicutes [meta_mOTU_v2_5720],Clostridium sp. CAG:568 [meta_mOTU_v2_5729],Butyricicoccus sp. BB10 [meta_mOTU_v2_5734],unknown Clostridiales [meta_mOTU_v2_5735],unknown Firmicutes [meta_mOTU_v2_5740],unknown Clostridiales [meta_mOTU_v2_5741],unknown Peptostreptococcaceae [meta_mOTU_v2_5742],unknown Clostridiales [meta_mOTU_v2_5745],unknown Clostridium [meta_mOTU_v2_5748],unknown Clostridium [meta_mOTU_v2_5751],unknown Clostridiales [meta_mOTU_v2_5754],Prevotella sp. CAG:485 [meta_mOTU_v2_5757],unknown Bacteroidales [meta_mOTU_v2_5773],Clostridium sp. CAG:288 [meta_mOTU_v2_5776],unknown Faecalibacterium [meta_mOTU_v2_5779],unknown Prevotella [meta_mOTU_v2_5780],unknown Clostridiales [meta_mOTU_v2_5783],unknown Clostridiales [meta_mOTU_v2_5791],unknown Butyricicoccus [meta_mOTU_v2_5800],unknown Clostridiales [meta_mOTU_v2_5805],unknown Clostridiales [meta_mOTU_v2_5806],unknown Veillonella [meta_mOTU_v2_5811],unknown Faecalibacterium [meta_mOTU_v2_5815],unknown Clostridiales [meta_mOTU_v2_5820],unknown Clostridiales [meta_mOTU_v2_5826],unknown Firmicutes [meta_mOTU_v2_5841],unknown Clostridiales [meta_mOTU_v2_5843],unknown Oscillibacter [meta_mOTU_v2_5845],unknown Clostridiales [meta_mOTU_v2_5849],unknown Bacteroidales [meta_mOTU_v2_5852],unknown Sutterella [meta_mOTU_v2_5857],unknown Dialister [meta_mOTU_v2_5867],Oscillibacter sp. CAG:155 [meta_mOTU_v2_5868],Clostridium sp. CAG:524 [meta_mOTU_v2_5874],unknown Clostridiales [meta_mOTU_v2_5880],unknown Roseburia [meta_mOTU_v2_5883],unknown Clostridiales [meta_mOTU_v2_5890],unknown Clostridiales [meta_mOTU_v2_5894],Prevotella sp. CAG:1092 [meta_mOTU_v2_5903],unknown Clostridiales [meta_mOTU_v2_5904],unknown Clostridiales [meta_mOTU_v2_5905],uncultured Clostridium sp. [meta_mOTU_v2_5907],Clostridium sp. CAG:451 [meta_mOTU_v2_5908],unknown Firmicutes [meta_mOTU_v2_5910],Clostridium sp. CAG:780 [meta_mOTU_v2_5914],unknown Clostridium [meta_mOTU_v2_5915],unknown Clostridiales [meta_mOTU_v2_5922],unknown Clostridium [meta_mOTU_v2_5927],unknown Sutterella [meta_mOTU_v2_5929],unknown Bacilli [meta_mOTU_v2_5934],unknown Clostridium [meta_mOTU_v2_5940],unknown Oscillibacter [meta_mOTU_v2_5944],unknown Tyzzerella [meta_mOTU_v2_5947],unknown Bacteroidales [meta_mOTU_v2_5951],unknown Clostridiales [meta_mOTU_v2_5954],unknown Clostridiales [meta_mOTU_v2_5959],unknown Clostridiales [meta_mOTU_v2_5961],Ruminococcus sp. CAG:254 [meta_mOTU_v2_5967],Eggerthella sp. CAG:298 [meta_mOTU_v2_5975],unknown Clostridiales [meta_mOTU_v2_5978],unknown Sutterella [meta_mOTU_v2_5982],unknown Clostridium [meta_mOTU_v2_5983],unknown Bacteroidales [meta_mOTU_v2_5989],unknown Clostridiales [meta_mOTU_v2_5991],unknown Desulfovibrio [meta_mOTU_v2_5993],unknown Clostridiales [meta_mOTU_v2_6000],Clostridium sp. CAG:465 [meta_mOTU_v2_6001],unknown Clostridiales [meta_mOTU_v2_6009],unknown Clostridiales [meta_mOTU_v2_6011],unknown Clostridiales [meta_mOTU_v2_6013],unknown Clostridiales [meta_mOTU_v2_6022],unknown Clostridiales [meta_mOTU_v2_6028],unknown Prevotella [meta_mOTU_v2_6029],unknown Clostridiales [meta_mOTU_v2_6036],unknown Clostridium [meta_mOTU_v2_6039],unknown Clostridiales [meta_mOTU_v2_6044],unknown Clostridiales [meta_mOTU_v2_6049],Anaeromassilibacillus sp. An200 [meta_mOTU_v2_6051],unknown Eggerthellales [meta_mOTU_v2_6052],unknown Clostridiales [meta_mOTU_v2_6054],unknown Verrucomicrobia [meta_mOTU_v2_6061],unknown Acetobacter [meta_mOTU_v2_6063],unknown Alistipes [meta_mOTU_v2_6070],unknown Parabacteroides [meta_mOTU_v2_6071],unknown Clostridiales [meta_mOTU_v2_6073],unknown Bacteria [meta_mOTU_v2_6079],unknown Clostridiales [meta_mOTU_v2_6080],unknown Bacteria [meta_mOTU_v2_6087],unknown Clostridiales [meta_mOTU_v2_6088],Verrucomicrobia bacterium CAG:312_58_20 [meta_mOTU_v2_6090],unknown Firmicutes [meta_mOTU_v2_6091],unknown Clostridium [meta_mOTU_v2_6098],unknown Clostridiales [meta_mOTU_v2_6105],unknown Clostridiales [meta_mOTU_v2_6107],unknown Pseudoflavonifractor [meta_mOTU_v2_6108],unknown Clostridium [meta_mOTU_v2_6117],Clostridium sp. CAG:533 [meta_mOTU_v2_6119],unknown Clostridiales [meta_mOTU_v2_6128],Clostridium sp. CAG:964 [meta_mOTU_v2_6131],unknown Clostridiales [meta_mOTU_v2_6134],unknown Clostridiales [meta_mOTU_v2_6144],unknown Clostridium [meta_mOTU_v2_6147],unknown Ruminococcus [meta_mOTU_v2_6152],Alistipes sp. An31A [meta_mOTU_v2_6154],unknown Clostridium [meta_mOTU_v2_6156],Clostridium sp. CAG:288 [meta_mOTU_v2_6160],unknown Bacteroidales [meta_mOTU_v2_6162],unknown Ruminococcaceae [meta_mOTU_v2_6174],unknown Roseburia [meta_mOTU_v2_6176],Clostridium sp. CAG:510 [meta_mOTU_v2_6178],unknown Clostridiales [meta_mOTU_v2_6190],unknown Clostridiales Family XIII. Incertae Sedis [meta_mOTU_v2_6192],unknown Clostridiales [meta_mOTU_v2_6193],unknown Bacteroidales [meta_mOTU_v2_6194],unknown Clostridiales [meta_mOTU_v2_6197],uncultured Eubacterium sp. [meta_mOTU_v2_6201],unknown Bacteroidales [meta_mOTU_v2_6217],Clostridium sp. CAG:343 [meta_mOTU_v2_6218],Clostridium sp. CAG:533 [meta_mOTU_v2_6224],unknown Firmicutes [meta_mOTU_v2_6228],unknown Clostridiales [meta_mOTU_v2_6233],unknown Firmicutes [meta_mOTU_v2_6237],unknown Clostridiales [meta_mOTU_v2_6238],unknown Clostridiales [meta_mOTU_v2_6249],unknown Alistipes [meta_mOTU_v2_6252],unknown Coprococcus [meta_mOTU_v2_6257],unknown Clostridiales [meta_mOTU_v2_6259],unknown Clostridiales [meta_mOTU_v2_6260],unknown Clostridiales [meta_mOTU_v2_6261],unknown Dehalococcoidales [meta_mOTU_v2_6270],unknown Clostridiales [meta_mOTU_v2_6272],unknown Clostridiales [meta_mOTU_v2_6275],unknown Akkermansia [meta_mOTU_v2_6276],unknown Firmicutes [meta_mOTU_v2_6277],unknown Clostridiales [meta_mOTU_v2_6281],unknown Clostridium [meta_mOTU_v2_6285],unknown Lactobacillales [meta_mOTU_v2_6288],unknown Prevotella [meta_mOTU_v2_6289],unknown Clostridium [meta_mOTU_v2_6292],unknown Prevotella [meta_mOTU_v2_6294],unknown Firmicutes [meta_mOTU_v2_6303],Eggerthella sp. CAG:368 [meta_mOTU_v2_6312],Brachyspira sp. CAG:484 [meta_mOTU_v2_6314],unknown Clostridiales [meta_mOTU_v2_6316],unknown Firmicutes [meta_mOTU_v2_6331],Roseburia sp. CAG:182 [meta_mOTU_v2_6334],unknown Dehalococcoidales [meta_mOTU_v2_6336],unknown Clostridium [meta_mOTU_v2_6338],Merdibacter massiliensis [meta_mOTU_v2_6340],unknown Clostridiales [meta_mOTU_v2_6348],unknown Clostridium [meta_mOTU_v2_6349],unknown Clostridium [meta_mOTU_v2_6363],unknown Clostridiales [meta_mOTU_v2_6371],unknown Bacteroidales [meta_mOTU_v2_6375],uncultured Clostridium sp. [meta_mOTU_v2_6381],unknown Prevotella [meta_mOTU_v2_6387],Coprobacillus sp. CAG:826 [meta_mOTU_v2_6390],Sutterella sp. CAG:521 [meta_mOTU_v2_6395],unknown Prevotella [meta_mOTU_v2_6399],Dialister invisus [meta_mOTU_v2_6402],Mycoplasma sp. CAG:472 [meta_mOTU_v2_6403],unknown Clostridiales [meta_mOTU_v2_6407],Firmicutes bacterium ADurb.Bin467 [meta_mOTU_v2_6414],unknown Prevotella [meta_mOTU_v2_6415],unknown Clostridiales [meta_mOTU_v2_6416],unknown Clostridiales [meta_mOTU_v2_6419],unknown Prevotella [meta_mOTU_v2_6430],unknown Eggerthellaceae [meta_mOTU_v2_6437],unknown Oscillibacter [meta_mOTU_v2_6438],unknown Faecalibacterium [meta_mOTU_v2_6452],unknown Prevotella [meta_mOTU_v2_6456],unknown Atopobiaceae [meta_mOTU_v2_6458],unknown Clostridiales [meta_mOTU_v2_6465],Clostridium sp. CAG:609 [meta_mOTU_v2_6466],Holdemanella biformis [meta_mOTU_v2_6468],unknown Clostridiales [meta_mOTU_v2_6475],Clostridium sp. CAG:1219 [meta_mOTU_v2_6476],unknown Ruminococcaceae [meta_mOTU_v2_6478],unknown Clostridiales [meta_mOTU_v2_6484],unknown Porphyromonas [meta_mOTU_v2_6490],unknown Clostridiales [meta_mOTU_v2_6495],unknown Bacteroidales [meta_mOTU_v2_6502],unknown Clostridiales [meta_mOTU_v2_6503],unknown Eubacterium [meta_mOTU_v2_6509],unknown Clostridiales [meta_mOTU_v2_6511],unknown Prevotella [meta_mOTU_v2_6516],Lentisphaerae bacterium ADurb.Bin242 [meta_mOTU_v2_6522],unknown Clostridiales [meta_mOTU_v2_6523],unknown Clostridiales [meta_mOTU_v2_6525],unknown Azospirillum [meta_mOTU_v2_6527],unknown Bacteroidales [meta_mOTU_v2_6528],Ruminococcus sp. CAG:382 [meta_mOTU_v2_6530],Coprobacillus sp. CAG:826 [meta_mOTU_v2_6536],unknown Clostridiales [meta_mOTU_v2_6548],unknown Ruminococcaceae [meta_mOTU_v2_6557],Clostridium sp. CAG:628 [meta_mOTU_v2_6559],unknown Clostridiales [meta_mOTU_v2_6561],unknown Clostridiales [meta_mOTU_v2_6571],unknown Clostridiales [meta_mOTU_v2_6575],unknown Clostridiales [meta_mOTU_v2_6585],unknown Bacteroidales [meta_mOTU_v2_6591],unknown Firmicutes [meta_mOTU_v2_6595],unknown Firmicutes [meta_mOTU_v2_6596],unknown Clostridiales [meta_mOTU_v2_6602],unknown Lachnospiraceae [meta_mOTU_v2_6615],unknown Atopobiaceae [meta_mOTU_v2_6622],unknown Lachnospiraceae [meta_mOTU_v2_6625],unknown Clostridiales [meta_mOTU_v2_6629],unknown Faecalibacterium [meta_mOTU_v2_6631],unknown Clostridiales [meta_mOTU_v2_6632],unknown Clostridiales [meta_mOTU_v2_6647],unknown Azospirillum [meta_mOTU_v2_6649],unknown Ruminococcaceae [meta_mOTU_v2_6652],unknown Eubacterium [meta_mOTU_v2_6657],unknown Prevotella [meta_mOTU_v2_6663],Ruminococcus sp. CAG:177 [meta_mOTU_v2_6664],unknown Bacteroidales [meta_mOTU_v2_6670],unknown Clostridiales [meta_mOTU_v2_6672],Oscillibacter sp. 57_20 [meta_mOTU_v2_6676],Mycoplasma sp. CAG:877 [meta_mOTU_v2_6682],unknown Clostridiales [meta_mOTU_v2_6686],unknown Methanomicrobia [meta_mOTU_v2_6693],unknown Prevotella [meta_mOTU_v2_6697],unknown Clostridiales [meta_mOTU_v2_6699],unknown Clostridiales [meta_mOTU_v2_6700],unknown Clostridiales [meta_mOTU_v2_6704],Bacillus sp. CAG:988 [meta_mOTU_v2_6706],unknown Firmicutes [meta_mOTU_v2_6711],unknown Clostridiales [meta_mOTU_v2_6716],unknown Clostridiales [meta_mOTU_v2_6721],Roseburia sp. CAG:303 [meta_mOTU_v2_6722],unknown Ruminococcus [meta_mOTU_v2_6724],unknown Ruminococcus [meta_mOTU_v2_6727],unknown Firmicutes [meta_mOTU_v2_6730],Clostridium sp. CAG:762 [meta_mOTU_v2_6739],unknown Clostridium [meta_mOTU_v2_6741],Clostridium sp. CAG:798 [meta_mOTU_v2_6742],unknown Clostridiales [meta_mOTU_v2_6751],unknown Firmicutes [meta_mOTU_v2_6758],unknown Clostridiales [meta_mOTU_v2_6760],unknown Veillonellaceae [meta_mOTU_v2_6765],Subdoligranulum sp. CAG:314 [meta_mOTU_v2_6768],unknown Flavobacteriia [meta_mOTU_v2_6771],unknown Collinsella [meta_mOTU_v2_6772],Clostridium sp. CAG:914 [meta_mOTU_v2_6776],unknown Clostridiales [meta_mOTU_v2_6777],Clostridium sp. CAG:1024 [meta_mOTU_v2_6786],unknown Clostridiales [meta_mOTU_v2_6787],Clostridium sp. CAG:710 [meta_mOTU_v2_6788],unknown Ruminococcaceae [meta_mOTU_v2_6789],unknown Clostridiales [meta_mOTU_v2_6791],unknown Clostridium [meta_mOTU_v2_6792],Roseburia sp. CAG:309 [meta_mOTU_v2_6793],unknown Clostridiales [meta_mOTU_v2_6795],unknown Clostridiales [meta_mOTU_v2_6801],unknown Clostridiaceae [meta_mOTU_v2_6802],unknown Clostridiales [meta_mOTU_v2_6807],unknown Clostridiales [meta_mOTU_v2_6808],unknown Eggerthellaceae [meta_mOTU_v2_6813],unknown Clostridiales [meta_mOTU_v2_6814],Dialister sp. CAG:357 [meta_mOTU_v2_6815],unknown Clostridium [meta_mOTU_v2_6816],Burkholderiales bacterium YL45 [meta_mOTU_v2_6818],unknown Clostridiales [meta_mOTU_v2_6819],unknown Dehalococcoidales [meta_mOTU_v2_6821],unknown Clostridiales [meta_mOTU_v2_6823],unknown Clostridiales [meta_mOTU_v2_6832],Prevotella sp. CAG:617 [meta_mOTU_v2_6833],unknown Clostridiales [meta_mOTU_v2_6834],unknown Anaerotruncus [meta_mOTU_v2_6835],unknown Firmicutes [meta_mOTU_v2_6848],unknown Ruminococcaceae [meta_mOTU_v2_6850],unknown Clostridiales [meta_mOTU_v2_6852],unknown Clostridiales [meta_mOTU_v2_6856],unknown Pasteurellaceae [meta_mOTU_v2_6865],unknown Clostridiales [meta_mOTU_v2_6867],Roseburia sp. 40_7 [meta_mOTU_v2_6875],unknown Clostridium [meta_mOTU_v2_6877],unknown Clostridium [meta_mOTU_v2_6883],unknown Clostridiales [meta_mOTU_v2_6885],unknown Clostridiales [meta_mOTU_v2_6891],Clostridium sp. CAG:628 [meta_mOTU_v2_6892],unknown Bacteroidales [meta_mOTU_v2_6903],unknown Ruminococcaceae [meta_mOTU_v2_6905],unknown Firmicutes [meta_mOTU_v2_6906],unknown Firmicutes [meta_mOTU_v2_6909],unknown Prevotella [meta_mOTU_v2_6911],unknown Veillonellaceae [meta_mOTU_v2_6915],unknown Clostridiales [meta_mOTU_v2_6916],unknown Peptostreptococcaceae [meta_mOTU_v2_6922],unknown Clostridiales [meta_mOTU_v2_6926],unknown Clostridiales [meta_mOTU_v2_6929],unknown Firmicutes [meta_mOTU_v2_6936],unknown Lachnospiraceae [meta_mOTU_v2_6937],Prevotella sp. CAG:279 [meta_mOTU_v2_6938],Staphylococcus sp. CAG:324 [meta_mOTU_v2_6946],unknown Bacteroidales [meta_mOTU_v2_6949],unknown Clostridiales [meta_mOTU_v2_6961],unknown Clostridiales [meta_mOTU_v2_6975],unknown Lentisphaerae [meta_mOTU_v2_6979],unknown Clostridiales [meta_mOTU_v2_6986],unknown Synergistaceae [meta_mOTU_v2_6989],unknown Eggerthella [meta_mOTU_v2_6998],unknown Acinetobacter [meta_mOTU_v2_7007],unknown Massiliomicrobiota [meta_mOTU_v2_7010],unknown Ruminococcaceae [meta_mOTU_v2_7012],unknown Clostridiales [meta_mOTU_v2_7014],unknown Prevotella [meta_mOTU_v2_7015],unknown Prevotella [meta_mOTU_v2_7016],unknown Clostridiales [meta_mOTU_v2_7018],Clostridium sp. CAG:798 [meta_mOTU_v2_7020],unknown Clostridium [meta_mOTU_v2_7026],unknown Clostridiales [meta_mOTU_v2_7031],Prevotella sp. CAG:485 [meta_mOTU_v2_7045],unknown Ruminococcus [meta_mOTU_v2_7048],Prevotella sp. CAG:873 [meta_mOTU_v2_7050],unknown Sutterellaceae [meta_mOTU_v2_7053],Coraliomargarita sp. CAG:312 [meta_mOTU_v2_7055],Clostridium sp. CAG:167 [meta_mOTU_v2_7056],unknown Clostridiales [meta_mOTU_v2_7058],unknown Akkermansia [meta_mOTU_v2_7059],unknown Clostridiales [meta_mOTU_v2_7061],unknown Clostridiales [meta_mOTU_v2_7063],unknown Clostridiales [meta_mOTU_v2_7066],unknown Clostridiales [meta_mOTU_v2_7067],unknown Clostridiales [meta_mOTU_v2_7072],Clostridium sp. CAG:411 [meta_mOTU_v2_7074],unknown Clostridiales [meta_mOTU_v2_7076],unknown Eggerthella [meta_mOTU_v2_7082],unknown Clostridiales [meta_mOTU_v2_7083],unknown Clostridiales [meta_mOTU_v2_7087],Eubacterium sp. CAG:581 [meta_mOTU_v2_7088],unknown Bacteroidales [meta_mOTU_v2_7089],unknown Clostridiales [meta_mOTU_v2_7093],unknown Clostridiales [meta_mOTU_v2_7097],Prevotella sp. CAG:1031 [meta_mOTU_v2_7101],unknown Clostridiales [meta_mOTU_v2_7104],unknown Oscillibacter [meta_mOTU_v2_7111],unknown Eubacterium [meta_mOTU_v2_7116],Clostridium sp. CAG:452 [meta_mOTU_v2_7118],unknown Clostridium [meta_mOTU_v2_7123],unknown Clostridiales [meta_mOTU_v2_7124],unknown Clostridiales [meta_mOTU_v2_7130],unknown Clostridiales [meta_mOTU_v2_7138],Eubacterium sp. CAG:274 [meta_mOTU_v2_7140],unknown Faecalibacterium [meta_mOTU_v2_7143],unknown Clostridiales [meta_mOTU_v2_7145],unknown Clostridiales [meta_mOTU_v2_7148],unknown Clostridiales [meta_mOTU_v2_7149],Clostridium sp. CAG:413 [meta_mOTU_v2_7152],unknown Clostridiales [meta_mOTU_v2_7153],Faecalibacterium prausnitzii [meta_mOTU_v2_7154],unknown Clostridiales [meta_mOTU_v2_7156],unknown Clostridiales [meta_mOTU_v2_7157],unknown Clostridiales [meta_mOTU_v2_7158],unknown Ruminococcus [meta_mOTU_v2_7159],unknown Clostridiales [meta_mOTU_v2_7173],unknown Firmicutes [meta_mOTU_v2_7175],unknown Clostridiales [meta_mOTU_v2_7180],Clostridium sp. CAG:302 [meta_mOTU_v2_7183],unknown Clostridiales [meta_mOTU_v2_7186],Clostridium sp. CAG:226 [meta_mOTU_v2_7187],unknown Clostridiales [meta_mOTU_v2_7188],unknown Clostridiales [meta_mOTU_v2_7192],unknown Prevotella [meta_mOTU_v2_7196],unknown Clostridiales [meta_mOTU_v2_7200],unknown Clostridiales [meta_mOTU_v2_7202],unknown Prevotella [meta_mOTU_v2_7203],unknown Clostridiales [meta_mOTU_v2_7209],unknown Bacteroidaceae [meta_mOTU_v2_7210],unknown Clostridiales [meta_mOTU_v2_7230],unknown Clostridium [meta_mOTU_v2_7253],Clostridium sp. CAG:451 [meta_mOTU_v2_7262],Clostridium sp. AT4 [meta_mOTU_v2_7263],unknown Clostridium [meta_mOTU_v2_7266],unknown Alistipes [meta_mOTU_v2_7270],unknown Ruminococcaceae [meta_mOTU_v2_7271],unknown Ruminococcus [meta_mOTU_v2_7275],Succinivibrio dextrinosolvens [meta_mOTU_v2_7277],Staphylococcus sp. CAG:324 [meta_mOTU_v2_7279],unknown Clostridiales [meta_mOTU_v2_7281],Dialister invisus [meta_mOTU_v2_7291],unknown Clostridiales [meta_mOTU_v2_7298],Sutterella sp. CAG:351 [meta_mOTU_v2_7305],unknown Clostridiales [meta_mOTU_v2_7306],unknown Alistipes [meta_mOTU_v2_7311],unknown Bacteroidales [meta_mOTU_v2_7313],unknown Clostridiales [meta_mOTU_v2_7317],unknown Prevotellaceae [meta_mOTU_v2_7319],unknown Clostridiales [meta_mOTU_v2_7320],unknown Clostridiales [meta_mOTU_v2_7323],Bacteroides sp. CAG:144 [meta_mOTU_v2_7324],Eubacterium sp. CAG:156 [meta_mOTU_v2_7325],Holdemanella biformis [meta_mOTU_v2_7329],unknown Peptostreptococcaceae [meta_mOTU_v2_7331],unknown Clostridiales [meta_mOTU_v2_7337],unknown Prevotella [meta_mOTU_v2_7342],unknown Bacteroidales [meta_mOTU_v2_7353],unknown Clostridiales [meta_mOTU_v2_7355],unknown Clostridiales [meta_mOTU_v2_7356],unknown Clostridiales [meta_mOTU_v2_7359],unknown Firmicutes [meta_mOTU_v2_7361],unknown Fusobacterium [meta_mOTU_v2_7372],unknown Dehalococcoidales [meta_mOTU_v2_7373],unknown Ruminococcaceae [meta_mOTU_v2_7375],Clostridium sp. CAG:448 [meta_mOTU_v2_7377],unknown Clostridium [meta_mOTU_v2_7389],unknown Bacteroidales [meta_mOTU_v2_7394],unknown Lachnospiraceae [meta_mOTU_v2_7398],unknown Ruminococcaceae [meta_mOTU_v2_7401],unknown Firmicutes [meta_mOTU_v2_7403],Bacteroides sp. 43_108 [meta_mOTU_v2_7407],unknown Clostridiales [meta_mOTU_v2_7415],unknown Firmicutes [meta_mOTU_v2_7419],unknown Tyzzerella [meta_mOTU_v2_7425],unknown Dehalococcoidales [meta_mOTU_v2_7432],unknown Burkholderiales [meta_mOTU_v2_7434],unknown Clostridiales [meta_mOTU_v2_7440],Eubacterium sp. CAG:202 [meta_mOTU_v2_7449],Clostridium sp. CAG:217 [meta_mOTU_v2_7451],unknown Firmicutes [meta_mOTU_v2_7454],unknown Clostridiales [meta_mOTU_v2_7455],unknown Clostridiales [meta_mOTU_v2_7462],unknown Clostridiales [meta_mOTU_v2_7467],unknown Clostridium [meta_mOTU_v2_7468],Anaerotruncus sp. CAG:528 [meta_mOTU_v2_7471],unknown Ruminococcus [meta_mOTU_v2_7476],unknown Clostridium [meta_mOTU_v2_7480],unknown Eggerthella [meta_mOTU_v2_7512],unknown Sutterella [meta_mOTU_v2_7526],unknown Clostridiales [meta_mOTU_v2_7527],unknown Clostridium [meta_mOTU_v2_7530],unknown Clostridiales [meta_mOTU_v2_7531],unknown Bacteroidaceae [meta_mOTU_v2_7534],unknown Clostridiales [meta_mOTU_v2_7541],unknown Clostridiales [meta_mOTU_v2_7546],unknown Clostridiales [meta_mOTU_v2_7550],unknown Clostridiales [meta_mOTU_v2_7553],unknown Clostridiales [meta_mOTU_v2_7561],unknown Roseburia [meta_mOTU_v2_7567],Clostridium sp. CAG:492 [meta_mOTU_v2_7568],unknown Collinsella [meta_mOTU_v2_7573],unknown Bacteroidaceae [meta_mOTU_v2_7579],unknown Bacteroidaceae [meta_mOTU_v2_7587],Holdemanella biformis [meta_mOTU_v2_7589],unknown Clostridiales [meta_mOTU_v2_7590],unknown Bacteroidaceae [meta_mOTU_v2_7591],unknown Ruminococcaceae [meta_mOTU_v2_7593],Mycoplasma sp. CAG:956 [meta_mOTU_v2_7596],unknown Clostridiales [meta_mOTU_v2_7600],unknown Azospirillum [meta_mOTU_v2_7608],Niameybacter massiliensis [meta_mOTU_v2_7610],Clostridium sp. CAG:1193 [meta_mOTU_v2_7613],Sutterella sp. CAG:351 [meta_mOTU_v2_7614],unknown Clostridiales [meta_mOTU_v2_7620],Coprobacillus sp. CAG:605 [meta_mOTU_v2_7625],Clostridium sp. CAG:433 [meta_mOTU_v2_7638],unknown Clostridiales [meta_mOTU_v2_7643],unknown Clostridium [meta_mOTU_v2_7645],unknown Pasteurellaceae [meta_mOTU_v2_7650],unknown Ruminococcaceae [meta_mOTU_v2_7652],unknown Porphyromonas [meta_mOTU_v2_7656],Sutterella sp. CAG:521 [meta_mOTU_v2_7660],Holdemanella biformis [meta_mOTU_v2_7667],Eubacterium sp. CAG:38 [meta_mOTU_v2_7668],unknown Azospirillum [meta_mOTU_v2_7674],unknown Clostridiales [meta_mOTU_v2_7682],unknown Clostridiales [meta_mOTU_v2_7684],unknown Clostridiales [meta_mOTU_v2_7685],unknown Eubacterium [meta_mOTU_v2_7687],unknown Firmicutes [meta_mOTU_v2_7689],unknown Eggerthella [meta_mOTU_v2_7693],unknown Clostridiales [meta_mOTU_v2_7697],unknown Clostridiales [meta_mOTU_v2_7702],unknown Clostridiales [meta_mOTU_v2_7707],unknown Clostridiales [meta_mOTU_v2_7717],unknown Faecalibacterium [meta_mOTU_v2_7718],unknown Clostridium [meta_mOTU_v2_7721],unknown Sutterella [meta_mOTU_v2_7723],unknown Olsenella [meta_mOTU_v2_7727],unknown Clostridiales [meta_mOTU_v2_7731],unknown Clostridiales [meta_mOTU_v2_7735],unknown Azospirillum [meta_mOTU_v2_7737],unknown Firmicutes [meta_mOTU_v2_7746],unknown Bacteroidales [meta_mOTU_v2_7748],unknown Clostridium [meta_mOTU_v2_7749],unknown Firmicutes [meta_mOTU_v2_7751],unknown Clostridiales [meta_mOTU_v2_7752],Ruminococcus sp. CAG:724 [meta_mOTU_v2_7753],unknown Firmicutes [meta_mOTU_v2_7755],unknown Firmicutes [meta_mOTU_v2_7760],Clostridium sp. CAG:138 [meta_mOTU_v2_7765],unknown Clostridiales [meta_mOTU_v2_7769],Staphylococcus sp. CAG:324 [meta_mOTU_v2_7772],Ruminococcus sp. CAG:403 [meta_mOTU_v2_7774],unknown Porphyromonas [meta_mOTU_v2_7777],unknown Clostridiales [meta_mOTU_v2_7778],unknown Clostridiales [meta_mOTU_v2_7781],unknown Clostridiales [meta_mOTU_v2_7782],unknown Clostridiales [meta_mOTU_v2_7784],Clostridium sp. CAG:230 [meta_mOTU_v2_7788],Clostridium sp. CAG:1193 [meta_mOTU_v2_7789],unknown Erysipelotrichaceae [meta_mOTU_v2_7790],unknown Clostridiales [meta_mOTU_v2_7795],unknown Clostridiales [meta_mOTU_v2_7800]"
    headerList = header.split(',')
    return headerList
def main():
    data = np.load('info.npz')
    usData = np.load("usInfo.npz")
    # print(data['e_data'].shape)
    # sdata, edata, cdata
    #s_train
    AllData = []
    AllData.append(clr(data["e_data"]))
    AllData.append(clr(data["c_data"]))
    X = usData["data"]
    goodx = copy.deepcopy(X[:,:93])

    Y = np.asarray(usData["y_train"])
    AllData.append(clr(goodx.T))

    AllY = []
    data2 = np.load('y_trains.npz')
    AllY.append(data2["e_train"])
    AllY.append(data2["c_train"])
    AllY.append(Y)


    # print(AllData[1].shape)
    #
    # print(AllData[2].shape)

    # print(AllY[2].shape)

    # print(AllY[1].shape)

    AnomolyDetector(AllData, AllY)
    RandForest(AllData, AllY)
    PooledRF(AllData,AllY)
    #
    # headers = readHeaders()
    # vals = featExtraction(AllData[2], AllY[2])
    # bubbleSort(vals, headers)
    # for x in range(len(headers)):
    #     print(headers[x] + "\t\t",vals[x])
    #     print()

main()
