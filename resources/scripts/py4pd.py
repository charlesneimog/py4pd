import pd

def pdprint(value):
    """Print a Pd Data type to the Pd console"""
    pd.print(str(value))


def py2pd(value):
    """Convert a Python data type to a PureData type"""
    return value


def pd2py(value):
    """Convert a PureData data type to a Python type"""
    return value

def nth(list, n):
    """Get the nth element of a list"""
    return list[n]


def pyiterate(value):
    """Iterate over a Python data type"""
    pd.iterate(value) 


def pylen(value):
    """Get the length of a Python data type"""
    return len(value)


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


def omsum(a, b):
    """ Sum two numbers or two lists of numbers """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x + b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a + x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x + y for x, y in zip(a, b)]
    else:
        pd.error("[py.sum]: bad arguments")


def omminus(a, b):
    """ Subtract two numbers or two lists of numbers """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x - b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a - x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x - y for x, y in zip(a, b)]
    else:
        pd.error("[py.minus]: bad arguments")


def omtimes(a, b):
    """ Multiply two numbers or two lists of numbers """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x * b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a * x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x * y for x, y in zip(a, b)]
    else:
        pd.error("[py.times]: bad arguments")


def omdiv(a, b):
    """ Divide two numbers or two lists of numbers """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a / b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x / b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a / x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x / y for x, y in zip(a, b)]
    else:
        pd.error("[py.div]: bad arguments")


def omabs(a):
    """ Absolute value of a number or a list of numbers """
    if isinstance(a, (int, float)):
        return abs(a)
    elif isinstance(a, list):
        return [abs(x) for x in a]
    else:
        pd.error("[py.abs]: bad arguments")


# =============================================
# =============================================
# =============================================

def py4pdLoadObjects():
    pd.addobject(pdprint, "py.print", no_outlet=True)
    pd.addobject(py2pd, "py2pd")
    pd.addobject(pd2py, "pd2py", pyout=True)

    # loop functions
    pd.addobject(pyiterate, "py.iterate")
    pd.addobject(pycollect, "py.collect")

    # list functions
    pd.addobject(pylen, "py.len")
    pd.addobject(nth, "py.nth")

    # ordinary functions
    pd.addobject(omsum, "py.sum")
    pd.addobject(omminus, "py.minus")
    pd.addobject(omtimes, "py.times")
    pd.addobject(omdiv, "py.div")
    pd.addobject(omabs, "py.abs")

