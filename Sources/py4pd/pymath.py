import math
import random

import pd

#  ========================
#  == LIST COMPREHENSION ==
#  ========================


def py4pdListComprehension(*a):
    """
    It uses the args of the objects as list comprehension.
    """
    listofArgs = pd.get_object_args()
    for i in range(len(a)):
        exec(f"in{i+1} = a[{i}]")
    list_comprehension_str = " ".join(map(str, listofArgs))
    try:
        # pass a as argument to eval
        return eval(list_comprehension_str)
    except Exception as e:
        pd.error(f"The list comprehension string is not valid: {e}")


#  ==============
#  Math Functions
#  ==============


def omsum(a, b=0):
    """Sum two numbers or two lists of numbers"""
    if isinstance(a, list) and isinstance(b, (int, float)):
        return list(map(lambda x: x + b, a))
    elif isinstance(a, (int, float)) and isinstance(b, list):
        return list(map(lambda x: x + a, b))
    elif isinstance(a, list) and isinstance(b, list):
        return [x + y for x, y in zip(a, b)]
    else:
        return a + b


def omminus(a, b):
    """Subtract two numbers or two lists of numbers"""
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
    """Multiply two numbers or two lists of numbers"""
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
    """Divide two numbers or two lists of numbers"""
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


def ommod(a, b):
    """Modulo two numbers or two lists of numbers"""

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a % b

    elif isinstance(a, list) and isinstance(b, (int, float)):
        return [x % b for x in a]

    elif isinstance(a, (int, float)) and isinstance(b, list):
        return [a % x for x in b]

    elif isinstance(a, list) and isinstance(b, list):
        return [x % y for x, y in zip(a, b)]
    else:
        pd.error("[py.div]: bad arguments")


def omabs(a):
    """Absolute value of a number or a list of numbers"""
    if isinstance(a, (int, float)):
        return abs(a)
    elif isinstance(a, list):
        return [abs(x) for x in a]
    else:
        pd.error("[py.abs]: bad arguments")


def pdround(f):
    return round(f)
