import pd

def splitSymbol(symbol):
    # return a list with all chars separated
    return list(symbol)

def py4pdLoadObjects():
    random = pd.new_object("py.splitSymbol")
    random.addmethod_symbol(splitSymbol)
    random.add_object()
