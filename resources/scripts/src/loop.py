import pd
import sys



def pyiterate(value):
    """
    if isinstance(value, list):
        pd.iterate(value)
        pd.out("output", out_n=1)
    else:
        pd.error("[py.iterate]: pyiterate only works with lists")
        return None
    """
    pd.out("py4pdcollect_clear", out_n=1)
    for i in value:
        pd.out(i)
    pd.out("py4pdcollect_output", out_n=1) 


def pycollect(data):
    """ It collects the data from pyiterate and outputs it """
    # lot of memory usage

    if str(data) == "py4pdcollect_output": 
        data = pd.getglobalvar("DATA")
        if data is None:
            pd.error("[py.collect]: There is no data to output")
        else:
            pd.out(data)

    elif str(data) == "py4pdcollect_clear":
        pd.print("[py.collect]: Clearing data")
        pd.clearglobalvar("DATA")

    else:
        olddata = pd.getglobalvar("DATA")
        if olddata is None:
            pd.setglobalvar("DATA", [data])
        else:
            pd.setglobalvar("DATA", olddata + [data])




def pytrigger(data):
    """ It triggers the pyiterate object """
    # get the amount of outlets
    outletCount = pd.get_out_count()
    for i in range(outletCount, -1, -1):
        pd.out(data, out_n=i)



def pyrecursive(data):
    """ It need to be used with pyiterate and pycollect in recursive patches """
    pd.recursive(data)
    


def pygate(value, out):
    """ It gates the output """
    pd.out(value, out_n=out)
