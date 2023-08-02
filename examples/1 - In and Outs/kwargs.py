import pd




def kwargs(**kwargs):
    for i in kwargs:
        pd.print(f"{i} = {kwargs[i]}")


def py4pdLoadObjects():
    pd.add_object(kwargs, "pykwargs")
