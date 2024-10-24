import pd
import random

def randomNumber(limit):
    return random.randint(0, limit) 

def py4pdLoadObjects():
    random = pd.new_object("py.floatRandom")
    random.addmethod_float(randomNumber)
    random.add_object()
