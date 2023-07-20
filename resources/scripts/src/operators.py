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

