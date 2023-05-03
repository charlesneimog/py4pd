import pd

def pyiterate(value):
    """Iterate over a Python data type"""
    if isinstance(value, list):
        pd.iterate(value)
    else:
        pd.error("[py.iterate]: pyiterate only works with lists")
        return None


def pycollect(data):
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
