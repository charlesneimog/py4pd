import pd


def pyand(*args):
    """Return the logical AND of all arguments."""
    return all(args)

def pyor(*args):
    """Return the logical OR of all arguments."""
    return any(args)

def pyequal(*args):
    """Return the logical equal of all arguments."""
    return all(x == args[0] for x in args)

def pygreater(x, y):
    """Return the logical greater of all arguments."""
    return x > y

def pylower(x, y):
    """Return the logical lower of all arguments."""
    return x < y

def py4pdif(condition, x, y):
    """Return the logical lower of all arguments."""
    if condition:
        return x
    else:
        return y

def pyisin(x, y):
    """Return the logical lower of all arguments."""
    return x in y
        
