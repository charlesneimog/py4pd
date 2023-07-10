import pd
import copy

def pyiterate(value):
    """Iterate over a Python data type"""
    # pd.out("clear", out_n=1)
    # NOTE: Old way
    """
    if isinstance(value, list):
        pd.iterate(value)
        pd.out("output", out_n=1)
    else:
        pd.error("[py.iterate]: pyiterate only works with lists")
        return None
    """
    for i in value:
        new_i = copy.deepcopy(i)
        pd.out(new_i)


    # pd.out("output", out_n=1)



def pycollect(data):
    """ It collects the data from pyiterate and outputs it """
    pointer = pd.getobjpointer()
    string = "py.collect_" + str(pointer)
    if data == "output":
        if string in globals():
            pd.out(globals()[string], pyiterate=True)
        else:
            pd.out(None)
    elif data == "clear":
        if string in globals():
            del globals()[string]
        else:
            pass
    else:
        if string in globals():
            if isinstance(globals()[string], list):
                globals()[string].append(data)
            else:
                globals()[string] = [globals()[string], data]
        else:
            globals()[string] = [data]
