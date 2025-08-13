# Creating Objects

As shown in the [Hello World](../hello) section, to create a new `py4pd` object you must use the subclass `puredata.NewObject`, define the object’s name, and save it in a folder using the pattern `<object_name>.pd_py`. To enable object creation, you must always import the `puredata` module, which is only available when the script `.pd_py` is loaded via `py4pd`.

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

#### `self.logpost`

Post things on PureData console.

#### `self.error`

Print error, same as logpost with error level.

#### `self.out`

Output data to the objecta.

#### `self.tabwrite`

Write `Tuple` of numbers in the `pd` array.

#### `self.tabread`

Read table from `pd`, returns a tuple.

#### `self.reload`

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

#### `self.new_clock`

`Clock` can be created using the `self.new_clock` method, which returns a `puredata.Clock` object. `new_clock` accepts a function as an argument, which will be executed when the clock ticks. In the above example, `self.metro` will have the methods: 

##### `delay`

Set a delay in milliseconds to execute function (in this case, `self.tick`).

##### `set`

TODO: 

##### `unset`

TODO: 

### Receivers

#### `self.new_receiver`

## Extra `puredata` Methods

#### `pd.post`

Post a message to PureData without being possible to detect the object (for example, for credits in objects).

#### `pd.hasgui` 

Returns if PureData has a GUI interface.
