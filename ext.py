import pd

def main():
    # try to import spam module
    try:
        import pd
        module = [1, 2, 3, 4, 5, 6]
        pd.out(module)
    except ImportError:
        print("Can't find spam module")
        module = "not ok"
   
    return None
