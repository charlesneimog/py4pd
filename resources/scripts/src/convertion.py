import pd


def py2pd(value):
    """Convert a Python data type to a PureData type"""
    return value

def pd2py(value):
    """Convert a PureData data type to a Python type"""
    return value


def pdlist2pylist(value):
    """Convert a PureData list to a Python list"""
    # value is one list, make it a string
    try:
        s = ''
        for i in range(len(value)):
            s = s + str(value[i]) + " " 
        s = s.replace(" ", ",")
        s = "[" + s + "]"
        lst = eval(s)
        return lst[0]
    except:
        pd.error("There is syntax error in the list")
        return None










