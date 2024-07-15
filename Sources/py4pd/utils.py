import pd


def getObjectArgs():
    pd.print(str(pd.get_object_args()))


def getObjectType(value):
    string = pd.get_py_type(value)
    pd.print("Object Type: " + string)
    return string
    


