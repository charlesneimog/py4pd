import pd

def mysum(x, y):
    return x + y

def mylistsum(x, y):
    x_sum = sum(x)
    y_sum = sum(y)
    return x_sum + y_sum

def mylib_setup():
    pd.add_object(mysum, "py.sum")
    pd.add_object(mylistsum, "py.listsum")
