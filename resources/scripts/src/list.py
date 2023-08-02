import pd
from typing import List


#  ==============
#  List Functions
#  ==============

def mat_trans(matrix: List[List]) -> List[List]:
    """
    Matrix transposition. The matrix is represented by a list of rows.
    Each row is a list of items. Rows and columns are interchanged.

    Args:
        matrix: A list of lists representing a matrix.

    Returns:
        A transposed matrix.
    """
    ## check if all rows have the same length
    if not all(len(row) == len(matrix[0]) for row in matrix):
        raise ValueError("All rows must have the same length")
    
    result = []
    max_len = max(map(len, matrix)) - 1
    for i in range(max_len + 1):
        row = [row[i] if i < len(row) else None for row in matrix]
        result.append(row)
    return result


def nth(list, n):
    """Get the nth element of a list"""
    if n is None or list is None:
        pd.error("[py.nth]: bad arguments")
        return None
    return list[n]


def omlist(*args):
    """Append a list to another list"""
    thelist = []
    for i in args:
        thelist.append(i)
    return thelist

def omappend(*args):
    """Append a list to another list"""
    # for example omappend([1,2,3], [4,5,6]) -> [1,2,3,4,5,6]
    thelist = []
    for i in args:
        if isinstance(i, list):
            thelist.extend(i)
        else:
            thelist.append(i)
    return thelist

def pylen(value):
    """Get the length of a Python data type"""
    return len(value)


def pymax(value):
    """Get the maximum value of a Python data type"""
    return max(value)


def pymin(value):
    """Get the minimum value of a Python data type"""
    return min(value)


def pyreduce(key, value):
    """Reduce a Python data type"""
    if value == [] or value == None:
        pd.error("[py.reduce]: bad arguments")
        return None

    if key == "+":  
        return sum(value)
    elif key == "-":
        result = 0
        for i in value:
            result = result - i
        return result
    elif key == "*":
        result = 1
        for i in value:
            result = result * i
        return result
    elif key == "/":
        result = 1
        for i in value:
            result = result / i
        return result
    else:
        pd.error("[py.reduce]: bad arguments")

def rotate(list, n):
    """Rotate a list"""
    if list is None:
        pd.error("[py.rotate]: bad arguments")
        return None
    return list[n:] + list[:n]


def flat(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flat(item))
        else:
            result.append(item)
    return result

