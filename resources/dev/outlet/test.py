import pd

def out():
    for i in range(110):
        pd.print(i % 5)
        pd.out(i, out_n=i % 5)



def py4pdLoadObjects():
    pd.addobject(out, 'py.out', num_aux_outlets=10) 

