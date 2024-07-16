import pd
import random

def randomNumber():
    return random.randint(0, 100)


def py4pdLoadObjects():
    objRandom = pd.new_object("py.random")
    objRandom.addmethod_bang(randomNumber)
    objRandom.add_object()
