# First Project

In this section is presented some _Hello World_ projects. One for each main way to work with PureData and Python.

## 1. Scripts Functions

The simplest way is to run your script using the py4pd object. In this, **you must add your script with the function you want to run in the same folder where your Pd patch is**. When you put a script file called `myscript.py` with a python function called `pdsum` inside you will run this function by creating a py4pd object with: `py4pd myscript pdsum`.

When you use this way to run your code, you must use the `run` method to run your python function. The function `pdsum` will be run by send a message `run 2 4` to an object, and it will output `6`.

```py
def pdsum(a, b):
    return a + b
```

!!! tip "About lists"

    To run function with list argument, you use the `[]` char. The python list `[1, 2, 3]` will be represented as `[1 2 3]`.

## 2. Simple Objects

If you don't want to use `run` method you can create object using the `pd.add_object` function. Then the function will be added as ordinary PureData objects. Let's say that you use a `mylib.py` in the same folder as your PureData patch. The object will be loaded using the `py4pd -lib mylib` object, and inside `mylib.py` you must add a function called `mylib_setup` (`[filename]_setup`) or `py4pdLoadObject`. For example:

```py
import pd

def pdsum(a, b):
    return a + b

def mylib_setup():
    # first arg is the function that will be executed
    # segund arg the name of the object
    pd.add_object(pdsum, 'pdsum')
```

With this script, after add an object `py4pd -lib mylib`, you can create the new object `pdsum`.

## 3. Complete Projects

With the previous way to create object, you will create an object that runs the same function for everything, i.e. `symbol`, `floats`, `lists`. The way presented here you can create functions for each one of these types, but also functions for specific `selectors`. For example, a function for single `float`, a function for `symbol`, a function for `lists`. For that we create our objects using the `pd.new_object` function.

```py
import pd

def bang_method():
    pd.print("Received a bang")

def float_method(a):
    return a + 50

def symbol_method(string):
    return f"received symbol is {string}"

def list_method(mylist):
    sum = 0
    for x in mylist:
        if type(x) == float or type(x) == int:
            sum += x
        else:
            pd.print(f"received symbol is {x}")
    return sum

def mymethod():
    pd.print("running my method")

def mylib_setup():
    newobj = pd.new_object('pdsum')
    newobj.addmethod_bang(bang_method)
    newobj.addmethod_float(float_method)
    newobj.addmethod_symbol(symbol_method)
    newobj.addmethod_list(list_method)
    newobj.addmethod_mymethod("mymethod", mymethod)

```

For complete projects, check the [new_object](../python-users/pd.new_object/methods/) docs.

## 4. Audio Objects

Sometimes, when you have some idea to a Pd Audio Object, you don't want to program `C` or `C++`, for that you can use the audio object with py4pd. The way of the creation is the same as presented in [Simple Objects](#2-simple-objects) and [Complete Objects](#3-complete-projects). The creation is very simple as this code (note that we have 3 audio object here):

```py
import pd

def sinusoids(freqs, amps):
    if freqs is None or amps is None:
        pd.error("You need to set the amplitudes")
        return np.zeros((1, 64))
    if len(freqs) != len(amps):
        pd.error("The length of freqs and amps are different")
        return None

    phases = pd.get_obj_var("PHASE", initial_value=np.zeros(len(freqs)))
    freqs = np.array(freqs, dtype=np.float64)
    amps = np.array(amps, dtype=np.float64)
    out, new_phases = mksenoide(freqs, amps, phases, 64, pd.get_sample_rate())
    # note that mksenoide is not defined here
    pd.set_obj_var("PHASE", new_phases)
    return out

def audioin(audio):
    num_channels = audio.shape[0]
    fftresult = []
    for channel in range(num_channels):
        channel_data = audio[channel, :]
        fft_result = np.fft.fft(channel_data)
        first_real_number = np.real(fft_result[0])
        fftresult.append(first_real_number)
    return fftresult

def audioInOut(audio):
    num_channels = audio.shape[0]
    transformed_audio = np.empty_like(audio, dtype=np.float64)

    for channel in range(num_channels):
        channel_data = audio[channel, :]
        fft_result = np.fft.fft(channel_data)
        ifft_result = np.fft.ifft(fft_result).real
        transformed_audio[channel, :] = ifft_result

    return transformed_audio


def py4pdLoadObjects():
    pd.add_object(sinusoids, "sinusoids~", obj_type=pd.AUDIOOUT)
    pd.add_object(audioin, "audioin~", obj_type=pd.AUDIOIN)
    pd.add_object(audioInOut, "audio~", obj_type=pd.AUDIO)

```

For complete docs check the [Audio Objects](../python-users/objtypes/audio/#audio-inout) docs.
