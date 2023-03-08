# Embedded Module

For those using Python, it is possible to communicate between Python and PureData using some of the functions in the Pd module. The Pd module is embedded in the py4pd code and accessible only in the PureData environment. It is similar to what is used inside Google Collab like `google.colab.drive`, `google.colab.widgets`, and others.

For example, to write to a PureData array you can use the method called `pd.tabwrite`, which accepts the array name and one list or numpy array and a keyword (`resize=`) where you resize or not the table. 

``` py
import pd
from random import randint

def pd_tabwrite():
    "It writes data on the pd.tabwrite array."
    randomNumbers = []
    tablen = randint(10, 200)
    i = 0
    while i < tablen:
        # gerar aleatoric number between -1 and 1
        randomNumbers.append(randint(-100, 100) / 100)
        i += 1
    pd.tabwrite("pd.tabwrite", randomNumbers, resize=True) # (1)!

```

1.  There should be an array called `pd.tabwrite` in Patch.


This will write the list `randomNumbers` in the `pd.tabwrite` table in PureData. If the table not exist it will give an error, like happens in tabwrite object.

# Table of Embedded Method

--------------------------- 
* [pd.out](###pd.out) - Output in PureData from any place in Python Code 
* [pd.send](###pd.send) - Send data to PureData, it is received with `receive` object.
* [pd.print](###pd.print) - Print in PureData console.
* [pd.tabwrite](###pd.tabwrite) - Write data in PureData arrays.
* [pd.tabread](###pd.tabread) - Read PureData arrays.
* [pd.show](###pd.show) show images in PureData canvas.
* [pd.home](###pd.home) - Get the current directory of the PureData Patch.
* [pd.tempfolder](###pd.tempfolder) - Get the tempfolder directory of `py4pd`. It's always clean.
* [pd.getkey](###pd.getkey) - Get keys saved with `key` message in `py4pd` object.
* [pd.samplerate](###pd.samplerate) - Get the current Sample Rate of PureData
* [pd.vecsize](###pd.vecsize) - Get current vector size of PureData.


# Methods description

-----------------------------------
## Changing Data

### `pd.out` 

With this method you can output things without the Python function finish the work (normally we return data for PureData using `return`). For example, given this function:

``` py
import pd


def example_pdout():
    for x in range(10):
    	pd.out(x)
    return x
```
it will output 1, 2, 3 (...) as it loops, similar what can be done with `else/iterate` or `cyclone/zl iter 1`. This can be used for `output` others information. For example, if I want to use the Partial Tracking provided by `loristrck`, it is possible use the `pd.out` to output the info about `frequency`, `amplitude` and `phrase` and save it on an `cyclone/coll` using the `time` as trigger. 

``` py

import pd
import loristrck


def lorisAnalisys(audiofile, parcialnumber):
    audiopathname = pd.home() + '/' + audiofile
    samples, sr = loristrck.util.sndreadmono(audiopathname)
    partials = loristrck.analyze(samples, sr, resolution=30, windowsize=40, 
                      hoptime=1/120)
    selected, noise = loristrck.util.select(partials, mindur=0.02, maxfreq=12000, 
                            minamp=-60,)

    parcialnumber = int(parcialnumber) # (1)!
    pdPartial = []
    for partial in selected:
        sec2ms = int(partial[parcialnumber][0] * 1000) # time
        try:
            pdPartial.append(sec2ms) 
            pdPartial.append(partial[parcialnumber][1]) # frequency
            pdPartial.append(partial[parcialnumber][2]) # amplitude
            pdPartial.append(partial[parcialnumber][3]) # prhase
            pd.out(pdPartial) # output data to save it in the cyclone/coll
            pdPartial = []
        except:
            pd.out([sec2ms, 0, 0, 0]) # if something go wrong
    
	pd.print("Done") # Inform the user that the process is finished

```

1.  PureData just have floats, in indices, we need to use `int` to convert the `float` received to an `int`.

<p align="center">
  <img src="https://raw.githubusercontent.com/charlesneimog/py4pd/develop/docs/assets/EXAMPLE-pd.out.png" alt="pd.out Example" width="50%">

<p align="center"><a href="https://github.com/charlesneimog/py4pd/raw/develop/docs/assets/EXAMPLE-pd.out.zip">Download</a></p>

</p>

---------------------------

### `pd.send` 

With `pd.send` you can send data for `receive` object in PureData Patch. It accepts two arguments, the `receive` name and the value that will be sent. For example, 
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

Let's say that you are using a lot of synths and have some color way to organize some combination of sounds. 

(...)

### `pd.tabwrite` 

As the name makes clear, it is a copy of the object `tabwrite`. 

### `pd.tabread`

As the name makes clear, it is a copy of the object `tabread`.

### `pd.getkey`

When you use audio with `py4pd`, I chose to make the function accept only one argument, the audio. This become the things more simple in a lot of senses. You must argue then: how to use different parameters inside audio functions, like `fft-size`, `bandwidth` and others. It will be done with `pd.getkey`. This function works together with the message `key` in PureData. Where the user set one name in the value. So, if the user send a message for `py4pd` using `key fft-size 1024`, you can get this value using `pd.getkey("fft-size")`. If the `key` was not defined by the user, the function return `None`, then you can set a default value.

------------------

## Info for the user

### `pd.print` 

`pd.print` must be used for inform the user some useful information. 

For example, when you are using the `py4pd`, you can debug your Python code from PureData. Note that, with PureData, print will not work! You need to use `pd.print`. Another example are this examples files, if you do not have, for example `loristrck` installed, I will use `pd.print` to inform what the user must do to the patch work.

### `pd.error` 

`pd.error` must be used for inform the user some errors inside Python. I am normally using it with `try` and `expect`. The difference between `pd.print` and `pd.error` is that the error is print in the `red`. 


## Images
 
### `pd.show`

`pd.show` is the copy of `else/pic` with an interface with Python. It can be used to show images inside PureData patches. It just work with `.gif`, `.ppm` and `.bmp`, but this is already enough to get, for example, Scores inside PureData. See this Python script with `neoscore`. This is a complete and useful way to show scores in PureData.

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
  <img src="https://raw.githubusercontent.com/charlesneimog/py4pd/develop/resources/examples/scores/score.gif" alt="pd.out Example" width="50%">
</p>

This script in delivered with `py4pd`, so if you create `py4pd` using `py4pd -score score chord` this will work.


## Files

### `pd.home`

`pd.home` is used to get the place where the patch is located. With this you can use `relative` paths for `py4pd` too.


### `pd.tempfolder`

`pd.tempfolder` is used get one tempfolder to save stuff that won't be used more than once. In `pd.show` I am using `pd.tempfolder`. All the data inside this folder will be deleted after you close PureData or delete all `py4pd` objects. The tempfolder is located in `~/.py4pd` for Windows, Linux and Mac and it is a hidden folder on Windows too.


## Audio
    
### `pd.samplerate`

This get the current samplerate of PureData.

### `pd.vecsize`

This get the current vectorsize/blocksize of PureData.



