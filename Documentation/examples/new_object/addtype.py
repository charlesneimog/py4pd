import pd

def receivedFloat(a):
    return "Received Python Float"

def receivedString(d):
    return "Received Python String"

def receivedList(f):
    return "Received Python List"   

def py4pdLoadObjects():
    random = pd.new_object("py.4types")
    random.addtype("float", receivedFloat)
    random.addtype("str", receivedString)
    random.addtype("list", receivedList)
    random.add_object()
