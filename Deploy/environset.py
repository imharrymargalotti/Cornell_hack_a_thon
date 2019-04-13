import os


def set_path_cow_data():
    os.environ["COW_CSV"] = str('/Users/timc/PycharmProjects/CowPY/Data/table.from_biom.txt')



'''
this is where we will use environment variables for paths and such

TO ASSIGN:
os.environ["variable_name"] = value

TO VIEW ALL:
print(os.environ)

to access: if the expected value is not there it likely is not set correctly and you should see a value of NONE in there.
var = os.environ.get('KEY_THAT_MIGHT_EXIST')
'''
