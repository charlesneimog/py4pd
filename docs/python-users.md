# Python Users

If you're using Python and PureData together, you can use the pd module within the py4pd package to exchange data, set configurations, and inform users of errors, among other things. This module is embedded in the `py4pd` code and is only accessible within the `py4pd` environment. It's similar to how Google Collab uses modules like `google.collab.drive` and `google.collab.widgets`. In the next section I present all the methods that are embbeded.

-------------------------------------- 
## <h2 style="text-align:center"> **Embbeded Module with `py4pd`** </h2>
--------------------------------------
### <h3 style="text-align:center"> **Write PureData Objects** </h3>


#### <h4 style="text-align:center"> `pd.addobject` </h4>

You can create your own objects with Python. For that, you define the Python Function and add it as an object using `#!python pd.addobject()`.

??? danger "Breaking Changes"
	I had change how `pd.addobject` work from version `0.6` to version `0.7`. Now, me use the function and the Pure Data object. Instead of use this, `pd.addobject("mysumObject", "NORMAL", "myNewPdObjects", "mysumObject")` we use this `pd.addobject(mysumObject, "mysumObject")`.

##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `Python Function` | Function that will be executed by the object.  |
| `arg2`   | `String` | String to create the object. |


##### <h5 style="text-align:center"> Keywords </h5> 


| Keyword     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `objtype`   | `pd` | The type of the object: `pd.VIS`, `pd.AUDIO`, `pd.AUDIOIN`, or `pd.AUDIOOUT`. Hiding this option will create a normal object.  |
| `figsize`   | `Tuple` | Sets the pixel size of the object. Example: `figsize=(400, 200)` creates an object with a width of 400 and height of 200. |
| `pyout`    | `Boolean` | Determines whether the output will be in PureData data types or Python. If set to Python, it cannot be used by PureData. |
| `no_outlet`    | `Boolean` | Creates an object with no outlets if set to `True`. |
| `added2pd_info`    | `Boolean` | Prints the message `"[py4pd]: Object {objectName} added to PureData"` when set to `True`. |


##### <h5 style="text-align:center"> Example </h5>

``` python

pd.addobject(myFunction, "mypyobj", 
            objtype=pd.VIS, figsize=(400, 200), 
            pyout=True, no_outlet=False, added2pd_info=False)

```

``` py

import pd


def mysumObject(a, b, c, d):
    return a + b + c + d

def py4pdLoadObjects():
    pd.addobject(mysumObject, "mysumObject")

    # My License, Name and University, others information
    pd.print("", show_prefix=False)
    pd.print("GPL3 2023, Your Name", show_prefix=False)
    pd.print("University of São Paulo", show_prefix=False)
    pd.print("", show_prefix=False)

```

Here we add the function `mysumObject` in PureData enviroment. For more infos read the [Python Objects](https://www.charlesneimog.com/py4pd/researchers/) page.

<p align="center">
    <img src="../examples/createobj/mynewpdobject.png" width="50%"</img>
</p>

-------------------------------------- 
### <h3 style="text-align:center"> **Exchanging Data** </h3>

These are the methods used to send data from Python to PureData. The inverse path is done mainly with `run` and `key` messages. 
<br>
--------------------------------------
#### <h4 style="text-align:center"> `pd.out` </h4>

`#!python pd.out()` allows you to output data to PureData without needing to wait for the Python function to finish executing. This is different from returning data to PureData using the `#!python return` statement, which requires the function to complete before sending data. 

##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `Python Object` | Python thing that will be outputed. |

##### <h5 style="text-align:center"> Keywords </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `symbol`   | `string` | It prepend the string in the output, can be used with `route` object. |

##### <h5 style="text-align:center"> Example </h5>

For example, consider the following function:

``` py
import pd


def example_pdout():
    for x in range(10):
    	pd.out(x, symbol="loop")
    pd.out("fim", symbol="end")

```

</p>

---------------------------

#### <h4 style="text-align:center"> `pd.send` </h4>

You can use `pd.send` to send data to a receive object in your PureData patch. This method takes in two arguments: the name of the `receive` object and the value you want to send. For instance, suppose you have a receive object named "myReceiver" in your patch. To send the value 42 to this object, you could use `pd.send("myReceiver", 42)`.

##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `string` | Name of the receive object. |
| `arg2`   | `Python Object` | Data that will be sent. |

##### <h5 style="text-align:center"> Example </h5> 

``` python
import pd


def pd_send():
    "It sends a message to the py4pdreceiver receive."	
	pd.send("py4pdreceiver", "hello from python!")
	pd.send("py4pdreceiver", 1) 
	pd.send("py4pdreceiver", [1, 2, 3, 4, 5])
	return 0

```

In this example, it will send to `py4pdreceiver` the message `"hello from python!"`, then the number `1`, then the list `[1, 2, 3, 4, 5]`. 

-------------------------------------- 
#### <h4 style="text-align:center"> `pd.tabwrite` </h4>

`pd.tabwrite` is a method that is essentially a copy of the `tabwrite` object in PureData. With this method, you can write audio or any data supported to PureData array.

##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `string` | Name of the table. |
| `arg2`   | `Python Object` | List or array (numpy) of numbers. |

##### <h5 style="text-align:center"> Keywords </h5> 

| Keyword     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `rezise`   | `Boolean` | Set if the table will be resized or not. |

##### <h5 style="text-align:center"> Example </h5> 

``` python

  pd.tabwrite("table2test", randomNumbers, resize=True)
  
```


-------------------------------------- 
#### <h4 style="text-align:center"> `pd.tabread` </h4>

`pd.tabread` is a method that is essentially a copy of the `tabread` object in PureData. With this method, you can read data from a PureData array directly from within your Python code. It will return one Numpy Array with the data of the table.

##### <h5 style="text-align:center"> Parameters </h5> 


| Parameters | Type    | Description                  |
| :--------: | :-----: | :--------------------------: |
|   `arg1`   | `string`|    Name of the table.        |



##### <h5 style="text-align:center"> Example </h5> 

``` py
import pd

def readFromArray():
    valuesFromArray = pd.tabread("py4pdArray")
    return valuesFromArray # This code don't make any sense :), but you understand it.

```

-------------------------------------- 
#### <h4 style="text-align:center"> `pd.getkey` </h4>

With `pd.getkey`, you can retrieve the value of a specific key that has been set by the user in a `key` message to `py4pd` objects. For example, if the user sends a key message to `py4pd` with the name "fft-size" and a value of 1024, you can retrieve this value in your Python code using `pd.getkey("fft-size")`. If the user hasn't defined a particular key, `pd.getkey` will return `None`, allowing you to set a default value if necessary.

##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `string` | Name of the key. |

##### <h5 style="text-align:center"> Example </h5> 

``` py
import pd

def someAudioFunction(audio): # (1)!
    fftvalue = pd.getkey("fft-size")
    if fftvalue is None:
        fftvalue = 2048 # default value for fft-size key.
    
    # Do something with the audio.
    
    return myGreatAudioManipulation

```

1. Remember, audio functions that run with `py4pd script myaudiofunction -audio` receive just one `arg` that is the audio. In Audio objects written in Python this is different.

------------------

### <h3 style="text-align:center"> **Info for the user** </h3>

There are two messages used to print info in the PureData console, they are `pd.print` and `pd.error`.
<br>

------------------
#### <h4 style="text-align:center"> `pd.print` </h4>

The ordinary function `#!python print()` will not work in `py4pd` (unless that you open PureData from the terminal). So if you want to debug or print some info from the PureData console you need to use `#!python pd.print`. 


##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `Python Object` | Thing to print |

##### <h5 style="text-align:center"> Keyword </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `show_prefix`   | `Python Object` | When `False` it remove the string "[Python]" from the begin of the message | 

##### <h5 style="text-align:center"> Example </h5> 

``` py
import pd
	
pd.print("ok") # It prints "[Python] ok"
pd.print("ok", show_prefix=False) # It prints "ok".

```

-------------------------------------- 

#### <h4 style="text-align:center"> `pd.error` </h4>

If you want to inform errors in PureData console use `pd.error` method. 


##### <h5 style="text-align:center"> Parameters </h5> 

| Parameters     | Type | Description                   | 
| :-----------: | :----: | :------------------------------: |
| `arg1`   | `string` | Message of the error. |


##### <h5 style="text-align:center"> Example </h5> 

``` python
import pd

def main(arg1):
    if isinstance(arg1, list):
        for i in range(1, 10):
            
        
    
    except:
        pd.error("This is a not valid operation")


```




-------------------------------------- 

### <h3 style="text-align:center"> **Utilities** </h3>


#### <h4 style="text-align:center"> `pd.getobjpointer` </h4>

When working with audio objects, there are situations where we require global variables or variables that retain their values across different runs. For instance, when creating a Python function to generate a sine wave, we may need a global variable for the phase in order to generate a continuous waveform. However, using Python Global Variables can be problematic when working with multiple objects, as all functions would modify the phase value, potentially overwriting it unintentionally. To address this issue, we introduced the pd.getobjpointer function, which returns a unique string representing the pointer of the C object. This string is unique and can be utilized in other contexts to locate and retrieve the desired global variable. 

--------------------------------------

#### <h4 style="text-align:center"> `pd.getglobalvar` </h4>

When working with audio objects, we have another helpful function called pd.getglobalvar. This function serves a similar purpose to pd.getobjpointer, but with a slight difference. Here, it creates the variable automatically if it doesn't exist yet. Let's take a look at an example. In the code snippet below, when we use pd.getglobalvar("PHASE"), it retrieves the value of the variable associated with the current running object. If the value hasn't been set yet, it will be initialized to 0.0.

``` python

phase = pd.getglobalvar("PHASE", initial_value=0.0)
        
```

--------------------------------------

#### <h4 style="text-align:center"> `pd.setglobalvar` </h4>

To set new values for the variable of the object we use `pd.setglobalvar`. It recives the name of the variable (string) and the new value (any Python Thing).

``` python
pd.setglobalvar("PHASE", phase)
```

--------------------------------------

### <h3 style="text-align:center"> **Images** </h3>
 
<br>

#### <h4 style="text-align:center"> `pd.show` </h4>

If you want to display images inside your PureData patches using Python, you can use the pd.show method. This method is essentially a copy of the else/pic object, but with an interface that allows you to easily show images from within your Python code.

One important thing to note is that pd.show only works with `.png`, `.gif`, and `.ppm` image formats. However, this is usually enough to work with a wide range of images and can be particularly useful when working with scores in PureData.

For example, you can use the `neoscore` Python library along with `pd.show` to display scores directly in your PureData patches. This provides a complete and useful way to work with scores in PureData, and can greatly enhance your ability to work with music and audio data in your patches.
Overall, `pd.show` provides a convenient way to display images from within your Python code, and can be a valuable tool when working with PureData.

``` py
import pd
from random import randint
import os
try:
    from neoscore.common import *
except Exception as e:
    pd.error(str(e))
    pd.error(
        "Please, run 'pip install neoscore -t ./py-modules' in the terminal from current folder")


def getpitchKey(pitch):
    note = {
        # natural
        'c': ['c', ''],
        'd': ['d', ''],
        'e': ['e', ''],
        'f': ['f', ''],
        'g': ['g', ''],
        'a': ['a', ''],
        'b': ['b', ''],
        # sharp
        'c#': ['c', 'accidentalSharp'],
        'd#': ['d', 'accidentalSharp'],
        'e#': ['e', 'accidentalSharp'],
        'f#': ['f', 'accidentalSharp'],
        'g#': ['g', 'accidentalSharp'],
        'a#': ['a', 'accidentalSharp'],
        'b#': ['b', 'accidentalSharp'],
        # flat
        'cb': ['c', 'accidentalFlat'],
        'db': ['d', 'accidentalFlat'],
        'eb': ['e', 'accidentalFlat'],
        'fb': ['f', 'accidentalFlat'],
        'gb': ['g', 'accidentalFlat'],
        'ab': ['a', 'accidentalFlat'],
        'bb': ['b', 'accidentalFlat'],
    }
    return note[pitch]


def chord(pitches):
    try:
        neoscore.shutdown()
    except BaseException:
        pass
    neoscore.setup()
    py4pdTMPfolder = pd.tempfolder()
    for file in py4pdTMPfolder:
        if file.endswith(".ppm"):
            try:
                os.remove(py4pdTMPfolder + "/" + file)
            except BaseException:
                pass
    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)
    staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
    bassClef = 'bass'
    Clef(ZERO, staffBaixo, bassClef)
    Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42),
              Brush(Color(0, 0, 0, 0)), Pen(thickness=Mm(0.5)))
    for pitch in pitches:
        # in pitch remove not number
        pitchWithoutNumber = pitch.replace(pitch[-1], '')
        pitchOctave = int(pitch[-1])
        pitchClass, accidental = getpitchKey(pitchWithoutNumber)
        note = [(pitchClass, accidental, pitchOctave)]
        if pitchOctave < 4:
            Chordrest(Mm(5), staffBaixo, note, (int(1), int(1)))
        else:
            Chordrest(Mm(5), staffSoprano, note, (int(1), int(1)))
    randomNumber = randint(1, 100)
    notePathName = py4pdTMPfolder + "/" + pitch + f"{randomNumber}.ppm"
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    if os.name == 'nt':
        notePathName = notePathName.replace("\\", "/")
    pd.show(notePathName) ## HERE THE pd.show
    return None

```

<p align="center">
  <img src="../examples/score/score.gif" alt="pd.out Example" width="50%">
</p>

To use this script, you can create a `py4pd` object environment with `py4pd -score score chord`. 

-------------------------------------- 
### <h3 style="text-align:center"> **File Management** </h3>
-------------------------------------- 

Next is presented `pd.home` and `pd.tempfolder`. These functions can be useful for managing files in PureData and Python.

-------------------------------------- 
#### <h4 style="text-align:center"> `pd.home` </h4>

In `py4pd`, `pd.home` is a function that returns the path to the directory where the currently-running PureData patch is located. This can be useful for accessing files and resources relative to the location of the patch. 

-------------------------------------- 


#### <h4 style="text-align:center"> `pd.tempfolder` </h4>

`pd.tempfolder` get one tempfolder to save stuff that won't be used more than once. In `pd.show` I am using `pd.tempfolder`. All the data inside this folder will be deleted after you close PureData or delete all `py4pd` objects. The tempfolder is located in `~/.py4pd` for Windows, Linux and Mac and it is a hidden folder on Windows too.

--------------------------------------

#### <h4 style="text-align:center"> `pd.py4pdfolder` </h4>

`pd.py4pdfolder` returns the folder where the binary of `py4pd` is located.

-------------------------------------- 

### <h3 style="text-align:center"> **Audio Info** </h3>
    
<br>
#### <h4 style="text-align:center"> `pd.samplerate` </h4>

This get the current samplerate of PureData. You can use the `pd.SAMPLERATE` variable too.

-------------------------------------- 

#### <h4 style="text-align:center"> `pd.vecsize` </h4>

This get the current vectorsize/blocksize of PureData. You can use the `pd.VECSIZE` variable too.



