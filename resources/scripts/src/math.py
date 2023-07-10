import pd

#  ==============
#  Math Functions
#  ==============

def omsum(a, b):
    """ Sum two numbers or two lists of numbers """
    if isinstance(a, list) and isinstance(b, (int, float)):
        return [x + b for x in a]
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a + x for x in b]
    elif isinstance(a, list) and isinstance(b, list):
        return [x + y for x, y in zip(a, b)]
    else:
        pd.print(f"{a} + {b}")
        return a + b


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
