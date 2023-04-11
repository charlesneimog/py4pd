import pd
import os
try:
    from pip._internal.cli.main import main as pipmain
    addpip = True
except Exception as e:
    pd.error("Error importing pip: " + str(e))
    addpip = False


def pipinstall(package):
    """Install a Python package from Pd"""
    if isinstance(package, list):
        pd.print('Installing ' + package[1] + ' , please wait...')
        folder = package[0]
        if folder == "local":
            folder = pd.home()

        elif folder == "global":
            folder = pd.py4pdfolder()

        else:
            pd.error("[py.install]: the first argument must be 'local' or 'global'")
            return None
        
        package = package[1]
    else:
        pd.error("[py.install]: bad arguments")
        return None
    home = pd.home()
    # If linux or macos
    if os.name == 'posix':
        try:
            pipmain(['install', '--target', f'{folder}/resources/py-modules', package, '--upgrade'])
            pd.print("Installed " + package)
            return None
        except Exception as e:
            pd.error(str(e))
            return None
    
    # If windows
    elif os.name == 'nt':
        import subprocess
        try:
            # change / to \\
            home = home.replace('/', '\\')  
            # install package from pip without opening a new window
            subprocess.run(f'pip install --target {home}\\py-modules {package} --upgrade', shell=True, check=True)
            pd.print("Installed " + package)
            pd.print("I recommend restart PureData...")       
        except Exception as e:
            pd.error(str(e))
            return None
    else:
        pd.error("Your OS is not supported")
        return None

# ================
#  Info Functions
#  ===============

def pdprint(value):
    """Print a Pd Data type to the Pd console"""
    pd.print(str(value))

#  =================
# Convertion Objects
#  =================

def py2pd(value):
    """Convert a Python data type to a PureData type"""
    return value

def pd2py(value):
    """Convert a PureData data type to a Python type"""
    return value

#  ==============
#  List Functions
#  ==============

def nth(list, n):
    """Get the nth element of a list"""
    if n is None or list is None:
        pd.error("[py.nth]: bad arguments")
        return None
    return list[n]

def omlist(*args):
    """Append a list to another list"""
    return list(args)


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

#  ==============
#  Loop Functions
#  ==============

def pyiterate(value):
    """Iterate over a Python data type"""
    pd.iterate(value) 

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

#  ==============
#  Math Functions
#  ==============

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
    # Pip install
    if addpip:
        pd.addobject(pipinstall, "py.pip")

    # info
    pd.addobject(pdprint, "py.print", no_outlet=True)
    
    # Convertion Objects
    pd.addobject(py2pd, "py2pd")
    pd.addobject(pd2py, "pd2py", pyout=True)

    # List Functions
    pd.addobject(pylen, "py.len")
    pd.addobject(nth, "py.nth", pyout=True)
    pd.addobject(omlist, "py.list", pyout=True)
    pd.addobject(pymax, "py.max")
    pd.addobject(pymin, "py.min")
    pd.addobject(pyreduce, "py.reduce", pyout=True)

    # Loop Functions
    pd.addobject(pyiterate, "py.iterate") # these are special objects, they don't have a pyout argument but output py data types
    pd.addobject(pycollect, "py.collect") # these are special objects, they don't have a pyout argument but output py data types

    # Math Functions
    pd.addobject(omsum, "py.sum")
    pd.addobject(omminus, "py.minus")
    pd.addobject(omtimes, "py.times")
    pd.addobject(omdiv, "py.div")
    pd.addobject(omabs, "py.abs")


