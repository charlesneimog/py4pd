import pd
import sys


def pyiterate(value):
    """
    It iterates the value and outputs it.
    """
    pd.out(pd.CLEARLOOP, out_n=1)
    for i in value:
        pd.out(i, out_n=0)
    pd.out(pd.OUTLOOP, out_n=1)


def pycollect(data):
    """ 
    It collects the data from pyiterate and outputs it. 
    """
    if data is pd.OUTLOOP:
        data = pd.get_obj_var("DATA")
        if data is None:
            pd.error("[py.collect]: There is no data to output")
        else:
            pd.out(data)
    elif data is pd.CLEARLOOP:
        pd.clear_obj_var("DATA")

    else:
        pd.accum_obj_var("DATA", data)




def pytrigger(data):
    """ It triggers the pyiterate object """
    # get the amount of outlets
    outletCount = pd.get_out_count()
    for i in range(outletCount, -1, -1):
        pd.out(data, out_n=i)


def pyrecursive(data):
    """ It need to be used with pyiterate and pycollect in recursive patches """
    sys.setrecursionlimit(1000000)

    pd._recursive(data)
    


def pygate(value, out):
    """ It gates the output """
    pd.out(value, out_n=out)
