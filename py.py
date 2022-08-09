def sum(x, y):
    "It sums two numbers."
    x = int(x)
    y = int(y)
    return x + y

def arithm_ser(begin, end, step):
    "It calculates the arithmetic series."
    list = []
    for x in range(begin, end, step):
        list.append(x)
    return list

def fibonacci(n):
    "Calculate the nth fibonacci number."
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def thread_test():
    "It tests the threading module. Just return the hour after 5 seconds."
    import time
    time.sleep(5)
    hour = time.strftime("%H")
    minute = time.strftime("%M")
    second = time.strftime("%S")
    return f'{hour}:{minute}:{second}'

