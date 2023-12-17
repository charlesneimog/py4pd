import numpy
import pd


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def exe():
    import sys

    python_executable_path = sys.executable
    return python_executable_path


def sum_of_primes(start, end):
    return sum(num for num in range(start, end + 1) if is_prime(num))

def floatmethod():
    pd.print("Hello from float method")

def symbolmethod():
    pd.print("Hello from symbol method")

def listmethod():
    pd.print("Hello from list method")
    
def anythingmethod():
    pd.print("Hello from anything method")   
    
def bangmethod():
    pd.print("Hello from bang method")
    
def selectormethod():
    pd.print("Hello from selector method")
    
def myclass():
    newobj = pd.new_object("myclass")
    newobj.addmethod_float(floatmethod)
    newobj.addmethod_symbol(symbolmethod)
    newobj.addmethod_list(listmethod)
    newobj.addmethod_anything(anythingmethod)
    newobj.addmethod_bang(bangmethod)
    newobj.addmethod("mymethod", selectormethod)
    newobj.addmethod("mymethod2", selectormethod, arg_types=(pd.A_FLOAT, pd.A_SYMBOL)) 
    newobj.add_object()



def mytest():
    sum_of_primes(1, 1000000)
    return "ok"
