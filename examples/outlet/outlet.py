import pd

def out():
    for i in range(5):
        pd.out(i, out_n=i % 5)


def personalidNumberOfOutlet(value, outletOut):
    pd.out(value, out_n=outletOut)


def py4pdLoadObjects():
    pd.addobject(out, 'py.out', num_aux_outlets=4) 
    pd.addobject(personalidNumberOfOutlet, 'py.setout', require_outlet_n=True)

