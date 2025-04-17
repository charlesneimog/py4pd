from mypackage.submodule import *
import numpy as np

def helloworld():
    pd.print("Hello World!")

def funcyfunctionwithnumpy():
    myarray = np.zeros(20)
    pd.print("I am using numpy")

def py4pdLoadObjects():
    pd.add_object(helloworld, "py.helloworld")
    pd.add_object(funcyfunctionwithnumpy, "py.numpy")
