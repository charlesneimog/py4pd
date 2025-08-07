import pd

def listmethod(list):
    newlist = [] 
    for x in list:
        newlist.append(x * x)
    return newlist

def py4pdLoadObjects():
    random = pd.new_object("py.listmultiplier")
    random.addmethod_list(listmethod)
    random.add_object()
