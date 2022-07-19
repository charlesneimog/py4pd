def function1(x, y):
    x = int(x)
    y = int(y)
    
    return x + y

def function2(x):
    return int(x)

def function3(x):
    print("I received the number: " + str(x))
    return x

def function4(x):
    for x in range(1, x, 1):
        x = x + 1
    return x

def function5(x):
    return f'O numero é {x}.'


def function6(x):
    list = []
    for x in range(1, x, 1):
        list.append(x)
    return list