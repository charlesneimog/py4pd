# Hello World

## Creating a `py4pd` Object

To define a new `py4pd` object, create a subclass of `puredata.NewObject`, name the object, and finally name the script with the name + `.pd_py`. For example, `pymetro.pd_py` for the object `pymetro`. Place the file in a place where Pd can find it. 

!!! Warning "Don't forget to load `py4pd` first"


### Example

```py
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro"  # Name of the Pure Data object

    def __init__(self, args):
        self.inlets = 2    # Number of inlets
        self.outlets = 1   # Number of outlets
```

### Key Points

The Python class name (e.g., `pymetro`) can be any valid class name. The name attribute determines the name of the object inside Pure Data. `self.inlets` and `self.outlets` define the number of inlets and outlets for the object. When loading this object in Pure Data, use the name attribute value (`pymetro` in this example) as the object name.


## Input and Output

### Input 

The input design is inspired by the mature `pd-lua` project. For methods, use the format `in_<inlet_number>_<method>`. For example, to execute code when a `float` is received on inlet 1, define a method called `in_1_float`. Pd provides predefined methods that do not require a custom selector: `bang`, `float`, `symbol`, `list`, and `anything`. You can also create custom selectors (prefixes); for instance, `in_1_mymethod` will be executed when the message `mymethod` is sent to inlet 1 of the object.

### Output

To produce output, use the method `self.out`. For example, `self.out(0, pd.SYMBOL, "test238")` sends the symbol `"test238"` to outlet 0. The second argument specifies the data type, which can be `pd.SYMBOL` or `pd.FLOAT`. To output a list, use `pd.LIST` instead.

### PyObject

`py4pd` also implements the `PyObject` message, which allows you to share Python data types between `py4pd` objects. This enables the transfer of class instances, NumPy arrays, and other Python objects that are not supported by Pure Data’s traditional data types.

## Metronome Example

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
        self.metro = self.new_clock(self.tick)
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
