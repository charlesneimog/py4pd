def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
    
    
def sum_of_primes(start, end):
    return sum(num for num in range(start, end + 1) if is_prime(num))  
    
def mytest():
    sum_of_primes(1, 1000000)
    return "ok"
