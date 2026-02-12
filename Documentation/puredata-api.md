# `puredata` Python API of `py4pd`

This page is a **reference** for what the C core exports in the `puredata` module and what methods exist on `puredata.NewObject`.

If you are writing `.pd_py` objects, you typically:

- `import puredata as pd`
- subclass `pd.NewObject`
- implement inlet methods like `in_1_float(...)`


The object below reproduces the behavior of the Pure Data `metro` object.

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

---

## Constants

### Output type constants

- `puredata.FLOAT` (Pd `A_FLOAT`)
- `puredata.SYMBOL` (Pd `A_SYMBOL`)
- `puredata.LIST` (Pd `A_GIMME`)
- `puredata.PYOBJECT` (output Python Objects for Pd)

This is used inside methods like `self.out` and `pd.send`.

### Inlets types constants

- `puredata.SIGNAL`, relative to `&s_signal`;
- `puredata.DATA`, relative to `&s_anything`;

Used to create objects for signal processing. Default is always `puredata.DATA`. For example:

``` python
import puredata as pd

class pydsp(pd.NewObject):
    name: str = "pydsp"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = (pd.SIGNAL, pd.SIGNAL, pd.DATA)
```

In this object, we have 2 `pd.DATA` *inlets* and 3 *outlets*. First 2 outlets are `pd.SIGNAL` (audio) and the last one is a `pd.DATA`.

---

## `puredata` methods

### `puredata.post`

* `puredata.post(*args) -> bool`

Posts a message to the Pd console (not rastreable), 

---

### `puredata.hasgui`

* `puredata.hasgui() -> bool`

Returns `True` when Pd is running with a GUI.

---

### `puredata.get_sample_rate`

* `puredata.get_sample_rate() -> int`

Returns Pd sample rate.

---

## Time inside py4pd

### `puredata.new_clock`

* `puredata.new_clock(owner, callback) -> puredata.clock`

Creates a Pd clock owned by `owner` (`self`). When it fires it calls `callback()`.

### `puredata.clock.delay`

Schedule `callback()` after `ms` milliseconds.

### `puredata.clock.unset`

* `puredata.clock.unset() -> bool`

Cancel any scheduled callback.

### `puredata.clock.set`

* `puredata.clock.set(time: float) -> bool`

Schedule clock for an absolute systime. `True` on sucess.


Check the use with clock `pymetro`

```python
    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.time = 1000
        self.metro = pd.new_clock(self, self.tick)
        self.args = args

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
```

---

## Receivers in py4pd

### `puredata.new_receiver`

* `puredata.new_receiver(owner, symbol: str, callback) -> puredata.receiver`

Binds to `symbol` (global Pd symbol) and forwards incoming messages to `callback`.

```python
    def __init__(self, args):
        self.inlets = 0
        self.outlets = 1
        self.receiver = pd.new_receiver(self, "myreceiver", self.tick)

    def tick(self, args):
        self.logpost(1, "Received something")
```

---

## Class: `puredata.NewObject`

!!! tip "Main class to create a python object with `py4pd`"

Your `.pd_py` object class must inherit from `puredata.NewObject`.

### Constructor

`puredata.NewObject`

The C core uses this to ensure the module-level registry exists. In practice, you just need to define a class that is inherit of this.

``` python
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro"  # Name of the Pure Data object

    def __init__(self, args):
        self.inlets = 2    # Number of inlets
        self.outlets = 1   # Number of outlets
```

---

### Methods (available on your object instance)

#### `self.out`

* `self.out(outlet: int, type: int, value) -> bool`

Send data to a Pd outlet.

- `type` must be one of: `puredata.FLOAT`, `puredata.SYMBOL`, `puredata.LIST`, `puredata.PYOBJECT`
- Outlet is 1-based.

!!! warning "`self.out` is 1-based"

Behavior:

- `FLOAT`: converts `value` to float and outputs a Pd float
- `SYMBOL`: converts `value` to string and outputs a Pd symbol
- `LIST`: requires a Python `list` of numbers/strings
- `PYOBJECT`: sends an internal pointer token so another py4pd object can receive the original Python object via `in_N_pyobj`

---

#### `self.tabread`

`self.tabread(name: str) -> tuple[float, ...]`

Read a Pd array (`garray`) into a Python tuple.

---

#### `self.tabwrite`

* `self.tabwrite(name: str, data: list|tuple, resize=False, redraw=True) -> bool`

Write Python numeric data into a Pd array.

- `resize=True` will resize the Pd array to match input length
- `redraw=False` skips GUI redraw (useful for performance)

---

#### `self.get_current_dir`

* `self.get_current_dir() -> str | None`

Returns the directory of the current canvas.

---

#### `self.reload`

* `self.reload() -> bool`

Reloads the current `.pd_py` source file and swaps the running class instance.

---

#### `self.logpost`

* `self.logpost(loglevel: int, *args, prefix=True) -> bool`

Posts a log message to the Pd console.

- If called from a non-main thread, the message is queued to the Pd main thread.
- `prefix=True` prints `[{object_name}]: ...`.

---

#### `self.error`

* `self.error(*args) -> bool`

Prints an error tagged with the object name.

---

## Inlet callback naming (Pd → Python)

The message dispatcher in `py4pd.c` looks for methods named:

- `in_{INLET}_{SELECTOR}`

Where `INLET` is 1-based.

Common selectors:

- `bang` → `in_1_bang(self)`
- `float` → `in_1_float(self, f)`
- `symbol` → `in_1_symbol(self, s)`
- `list` → `in_1_list(self, xs)`
- any other selector (e.g. `set`) → `in_1_set(self, args)`

To receive Python objects sent with `puredata.PYOBJECT` implement:

- `in_{INLET}_pyobj(self, obj)`
