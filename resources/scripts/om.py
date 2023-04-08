import pd

def omsum(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x + b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a + x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x + y for x, y in zip(a, b)]
    else:
        pd.error("omsum: bad arguments")

def omminus(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x - b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a - x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x - y for x, y in zip(a, b)]
    else:
        pd.error("omminus: bad arguments")

def omtimes(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x * b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a * x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x * y for x, y in zip(a, b)]
    else:
        pd.error("omtimes: bad arguments")

def omdiv(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a / b
    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x / b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a / x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x / y for x, y in zip(a, b)]
    else:
        pd.error("omdiv: bad arguments")

def omabs(a):
    if isinstance(a, (int, float)):
        return abs(a)
    elif isinstance(a, list):
        return [abs(x) for x in a]
    else:
        pd.error("omabs: bad arguments")


def py4pdLoadObjects():
    pd.addobject(omsum, "om+")
    pd.addobject(omminus, "om-")
    pd.addobject(omtimes, "om*")
    pd.addobject(omdiv, "om/")
    pd.addobject(omabs, "omabs")
