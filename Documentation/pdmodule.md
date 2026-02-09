# Creating Objects

As shown in the [Hello World](hello.md) section, to create a new `py4pd` object you must use the subclass `puredata.NewObject`, define the object’s name, and save it in a folder using the pattern `<object_name>.pd_py`. To enable object creation, you must always import the `puredata` module, which is only available when the script `.pd_py` is loaded via `py4pd`.

## `NewObject` Class

All PureData Python objects are created as subclasses from the base class `puredata.NewObject`. 

```python
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro" # object name, must be exactly the same as the file name without extension (pymetro.pd_py)

    def __init__(self, args):
        # Object initializer
        pass
```

### Object Attributes

From the class initializer (`__init__`), you need to define some object attributes. Like `self.inlets`, `self.outlets` and options attributes (like `clocks`, `receivers`, etc).

* `self.inlets`: Can be an `integer` (number of inlets) or a `Tuple` specifying inlet types (`puredata.SIGNAL` or `puredata.DATA`), where each element in the tuple defines the type of the inlet at the corresponding index.

* `self.outlets`: Can be an `integer` (number of outlets) or a `Tuple` specifying outlet types (`puredata.SIGNAL` or `puredata.DATA`), where each element in the tuple defines the type of the outlet at the corresponding index.

### `pd.NewObject` Methods 

* `self.logpost`: Post things on Pd console, `self,logpost(0, "This is a fatal error")` `self.logpost(1, "This is an error")`, `self.logpost(2, "This normal a log")`, `self.logpost(3, "This is a debug")`.
* `self.error`: Print error, same as `self.logpost` with error level 1.
* `self.out`: Output data to the object. `self.out(0, pd.FLOAT, 1)`, `self.out(0, pd.SYMBOL, "hello")`. `self.out(0, pd.PYOBJECT, [[1,2,3][4,5,6]])`. 
* `self.tabwrite`: Write `Tuple` of numbers in the `pd` array.
* `self.tabread`: Read table from `pd`, returns a tuple.
* `self.reload`: Reload the object.

### Clocks

``` python
class pymetro(pd.NewObject):
    name: str = "pymetro"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.metro = pd.new_clock(self, self.tick)
```

#### `pd.new_clock`

`Clock` can be created using the `pd.new_clock` method, which returns a `puredata.Clock` object. `new_clock` accepts the `self` of the class and a function as an argument, which will be executed when the clock ticks. In the above example, `self.metro` will have the methods: 

* `self.metro.delay`: Set a delay in milliseconds to execute function (in this case, `self.tick`).

### Receivers

class pyreceiver(pd.NewObject):
    name: str = "pyreceiver"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.receiver = pd.new_receiver(self, "pyreceiver", self.received)

* `self.receiver.unbind()` = This make the object not receive messages from the symbol `pyreceiver`.
* `self.receiver.bind()` = This make the object receive messages from the symbol `pyreceiver`.


## Extra `puredata` Methods

* `pd.post`: Post a message to Pd without being possible to detect the object (for example, for credits in objects), or warnings when some package is not installed.
* `pd.hasgui`: Returns if Pd has a GUI interface.
* `pd.get_sample_rate`: Return sample rate of Pd.




## Complet Examples

Here some examples of objects:

### Tradicional Objects

``` python
import puredata as pd


class pymetro(pd.NewObject):
    name: str = "pymetro"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.toggle = False
        if len(args) > 0:
            self.time = float(args[0])
        else:
            self.time = 1000
        self.metro = pd.new_clock(self, self.tick)
        self.args = args

    def in_2_float(self, f: float):
        self.time = f

    def in_1_float(self, f: float):
        if f:
            self.toggle = True
            self.tick()
        else:
            self.metro.unset()
            self.toggle = False

    def in_1_reload(self, args: list):
        self.reload()

    def tick(self):
        if self.toggle:
            self.metro.delay(self.time)
        self.out(0, pd.SYMBOL, "test238")
``` 

### Python Data Types on Pd

One of the great things that made me start `py4pd` was that I missed some data types. With `py4pd`, you can use any datatype supported by Python. This includes:

* **Numeric types:** `int`, `float`, `complex`
* **Sequence types:** `list`, `tuple`, `range`
* **Text type:** `str`
* **Set types:** `set`, `frozenset`
* **Mapping type:** `dict`
* **Boolean type:** `bool`
* **Binary types:** `bytes`, `bytearray`, `memoryview`

This flexibility allows you to integrate Python data structures directly into your Pd patches, making data manipulation and processing much easier and more powerful.

To use it, you must convert the Pd data to Python using `py.2py` or use the `pd.PYOBJECT` type when using `self.out` method.


#Here’s a completed section including your example with a concise explanation:

---

### Audio Objects

With `py4pd`, you can create custom audio objects (tilde objects, like `osc~`) entirely in Python. This allows you to define signal processing logic directly in Python, while Pd handles the audio routing.

Example:

```python
import puredata as pd
import math

class pytest_tilde(pd.NewObject):
    name: str = "pytest~"

    def __init__(self, args):
        self.inlets = pd.SIGNAL
        self.outlets = pd.SIGNAL
        self.phase = 0

    def perform(self, input):
        # this is executed in each block of audio
        blocksize = self.blocksize
        samplerate = self.samplerate

        out_buffer = []
        for i in range(blocksize):
            phase_increment = 2 * math.pi * input[0][i] / samplerate
            sample = math.sin(self.phase)
            out_buffer.append(sample)
            self.phase += phase_increment
            if self.phase > 2 * math.pi:
                self.phase -= 2 * math.pi
        return tuple(out_buffer)

    def dsp(self, sr, blocksize, inchans):
        # this is executed when you turn on the audio
        self.samplerate = sr
        self.blocksize = blocksize
        self.inchans = inchans
        return True
```
