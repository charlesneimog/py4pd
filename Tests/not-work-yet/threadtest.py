import numpy
import pd


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def exe():
    import sys

    python_executable_path = sys.executable
    return python_executable_path


def sum_of_primes(start, end):
    return sum(num for num in range(start, end + 1) if is_prime(num))


def floatmethod(x, y):
    return x + y


def symbolmethod(a):
    pd.print(f"Symbol method, I received {a} symbol")


def listmethod(a, c):
    pd.print(f"a = {a} | c = {c}")


def anythingmethod(a):
    pd.print(f"Hello from anything method | a = {a}")


def bangmethod():
    pd.print("Hello from bang method")


def selectormethod(a, b, c, d):
    pd.print("Hello from selector method")


def selectormethod2():
    pd.print("Hello from selector2 method")


def audiomethod(audio):
    audioout = numpy.multiply(audio, 0.5)
    return audioout


def threadtest_setup():
    patchZoom = pd.get_patch_zoom()
    scoreImage = None
    if patchZoom == 1:
        scoreImage = "./resources/score_nozoom.gif"
    elif patchZoom == 2:
        scoreImage = "./resources/score.gif"

    # Vis Obj
    newobjVis = pd.new_object("vis")
    newobjVis.addmethod_list(listmethod)
    newobjVis.addmethod_anything(anythingmethod)
    newobjVis.type = pd.VIS
    newobjVis.image = scoreImage
    newobjVis.fig_size = (250, 250)
    newobjVis.add_object()

    # Audioin Obj
    newobj = pd.new_object("audio~")
    newobj.addmethod_audio(audiomethod)
    newobj.type = pd.AUDIO
    newobj.add_object()
