import os


def set_paths():
    os.environ["COW_CSV"] = str('/Users/timc/PycharmProjects/CowPY/Data/table.from_biom.txt')
    os.environ["ERR"] = str("/Users/timc/PycharmProjects/CowPY/Data/ERR1.csv")
    os.environ["Y"] = str("/Users/timc/PycharmProjects/CowPY/Data/y_trainDone.csv")
    os.environ["CCSI"] = str("/Users/timc/PycharmProjects/CowPY/Data/CCSIdata.csv")
    os.environ["SAMEA"] = str("/Users/timc/PycharmProjects/CowPY/Data/SAMEAdata.csv")
    os.environ["SY"] =str("/Users/timc/PycharmProjects/CowPY/Data/SraRunTableSAMEA.csv")
    os.environ["CY"] = str("/Users/timc/PycharmProjects/CowPY/Data/SraRunTableCCIS.csv")




'''
this is where we will use environment variables for paths and such

TO ASSIGN:
os.environ["variable_name"] = value

TO VIEW ALL:
print(os.environ)

to access: if the expected value is not there it likely is not set correctly and you should see a value of NONE in there.
var = os.environ.get('KEY_THAT_MIGHT_EXIST')
'''
