from om_py import to_om


def function (x, y):
    x = int(x)
    y = int(y)
    
    return x + y

def test2om(x):
    
    list = [1, 2, 3, 4, 5]
    list.append(x)
    print(list)
    return to_om(list)

