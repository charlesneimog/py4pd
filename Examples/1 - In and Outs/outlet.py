import pd


def out():
    for i in range(5):
        pd.out(i, out_n=i % 5)


def personalidNumberOfOutlet(value, outletOut):
    pd.out(value, out_n=outletOut)


def outlet_setup():
    pd.add_object(out, "py.out", n_outlets=5)
    pd.add_object(personalidNumberOfOutlet, "py.setout", require_outlet_n=True)
