# Creating Objects

As shown in the [Hello World](../hello) section, to create a new `py4pd` object you must subclass `puredata.NewObject`, define the object’s name, and save it in a folder using the pattern `<object_name>.pd_py`. To enable object creation, you must always import the `puredata` module, which is only available when the script is loaded via `py4pd`, and then create a subclass of `puredata.NewObject`. 

## `NewObject` Class

All `py4pd` object are created as subclasses from the base class `puredata.NewObject`. 

```python
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro" # object name, must be exactly the same as the file name (pymetro.pd_py)

    def __init__(self, args):
        # Object initializer
        pass
```

### Object Attributes

From the class initializer (`__init__`), you need to define some object attributes. Like `self.inlets`, `self.outlets` and others. 

* `self.inlets`: Can be an `integer` or a `Tuple` specifying inlet types (`puredata.SIGNAL` or `puredata.DATA`), where each element in the tuple defines the type of the inlet at the corresponding index.

* `self.outlets`: Can be an `integer` or a `Tuple` specifying outlet types (`puredata.SIGNAL` or `puredata.DATA`), where each element in the tuple defines the type of the outlet at the corresponding index.

### Simple Methods

#### `self.NewObject.logpost`

Post things on PureData console.

#### `self.NewObject.error`

Print error, same as logpost with error level.

#### `self.NewObject.out`

Output data to the objecta.

#### `self.NewObject.tabwrite`

Write `Tuple` of numbers in the `pd` array.

#### `self.NewObject.tabread`

Read table from `pd`, returns a tuple.

#### `self.NewObject.reload`

Reload the object.

### Clocks

``` python
class pymetro(pd.NewObject):
    name: str = "pymetro"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.metro = self.new_clock(self.tick)
```

#### `pd.NewObject.new_clock`

`Clock` can be created using the `self.new_clock` method, which returns a `puredata.Clock` object. `new_clock` accepts a function as an argument, which will be executed when the clock ticks. In the above example, `self.metro` will have the methods: 

#### `pd.NewObject.Clock.delay`
#### `pd.NewObject.Clock.set`
#### `pd.NewObject.Clock.unset`

### Receivers

#### `pd.NewObject.new_receiver`

## Extra `puredata` Methods

#### `pd.post`

Post a message to PureData without being possible to detect the object (for example, for credits in objects).

#### `pd.hasgui` 

Returns if PureData has a GUI interface.
