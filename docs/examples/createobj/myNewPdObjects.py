import pd

def mysumObject(a, b, c, d):
    return a + b + c + d

def py4pdLoadObjects():
    pd.add_object(mysumObject, "mysumObject") # function, string with name of the object
    
    # My License, Name and University, others information
    pd.print("", show_prefix=False)
    pd.print("GPL3 | by Charles K. Neimog", show_prefix=False)
    pd.print("University of São Paulo", show_prefix=False)
    pd.print("", show_prefix=False)
