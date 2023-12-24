import pd

def portugues(a, b, c):
    return "Portuguese method"

def english(d, e):
    return "English method"

def german(f):
    return "German method"
    
def dutch():
    return "Dutch method"    
    
def py4pdLoadObjects():
    random = pd.new_object("py.methods")
    random.addmethod("casa", portugues)
    random.addmethod("home", english)
    random.addmethod("haus", german)
    random.addmethod("huis", dutch)
    random.add_object()
