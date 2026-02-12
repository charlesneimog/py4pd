# Creating Objects (Pure Data module)

This page documents the **actual Python API exported by py4pd's C core** (see `Sources/py4pd.c`).

If you want to create a new `py4pd` object, you write a Python file with the extension `.pd_py`, define a class that inherits from `puredata.NewObject`, and make sure the **class `name` matches the filename**.

---

## Quick start

Create a file `mymetro.pd_py`:

```python
import puredata as pd


class mymetro(pd.NewObject):
    name: str = "mymetro"  # must match the filename (mymetro.pd_py)

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.ms = float(args[0]) if len(args) else 250.0
        self.running = False
        self.clock = pd.new_clock(self, self._tick)

    def in_1_float(self, f: float):
        self.running = bool(f)
        if self.running:
            self._tick()
        else:
            self.clock.unset()

    def in_2_float(self, f: float):
        self.ms = float(f)

    def _tick(self):
        if self.running:
            self.clock.delay(self.ms)
            self.out(0, pd.SYMBOL, "tick")
```

---

## How Pd messages map to Python methods

`py4pd` routes incoming messages to methods using this naming convention:

- `in_{INLET}_{SELECTOR}`

Where:

- `INLET` is **1-based** (first inlet is `1`, second is `2`, ...)
- `SELECTOR` is the Pd selector name (`bang`, `float`, `symbol`, `list`, or any custom selector)

### Argument conversion

`py4pd` converts Pd atoms to Python values like this:

- `bang` → calls `in_N_bang(self)` (no arguments)
- `float` → calls `in_N_float(self, f: float)`
- `symbol` → calls `in_N_symbol(self, s: str)`
- `list` → calls `in_N_list(self, xs: list)`
  - numbers become `int` when possible, otherwise `float`
  - symbols become `str`
- Any other selector (example: `set`, `foo`, `bar`) → calls `in_N_set(self, args: list)`
  - the single argument is a Python `list` of the message atoms (same conversion rules)

### Methods must return `None`

If your method returns anything other than `None`, `py4pd` will print an error in Pd.
Use `self.out(...)` to output values.

---

## Inlets and outlets

In `__init__`, set `self.inlets` and `self.outlets`.

These attributes can be:

### 1) An integer (count)

```python
self.inlets = 2
self.outlets = 3
```

### 2) A single string (one inlet/outlet)

Use the module constants:

- `pd.SIGNAL` (string: `"signal"`)
- `pd.DATA` (string: `"anything"`)

Example:

```python
self.inlets = pd.SIGNAL
self.outlets = pd.SIGNAL
```

### 3) A tuple of strings (typed configuration)

```python
self.inlets = (pd.SIGNAL, pd.DATA, pd.DATA)
self.outlets = (pd.DATA, pd.SIGNAL)
```

---

## Output from Python to Pd: `self.out`

Use:

```python
self.out(outlet_index, type_constant, value)
```

Type constants exported by the module:

- `pd.FLOAT`
- `pd.SYMBOL`
- `pd.LIST`
- `pd.PYOBJECT`

Examples:

```python
self.out(0, pd.FLOAT, 1.5)
self.out(0, pd.SYMBOL, "hello")
self.out(0, pd.LIST, [1, 2, 3, "a"])
```

### Passing arbitrary Python objects (`pd.PYOBJECT`)

You can send arbitrary Python objects between `py4pd` objects using `pd.PYOBJECT`.

Sender:

```python
self.out(0, pd.PYOBJECT, {"a": 1, "b": 2})
```

Receiver: implement `in_N_pyobj`:

```python
def in_1_pyobj(self, obj):
    # obj is the exact Python object that was sent
    self.logpost(2, "Got:", obj)
```

---

## Arrays (Pd tables): `tabread` / `tabwrite`

Read an array as a Python tuple of floats:

```python
data = self.tabread("myarray")
```

Write a list/tuple of numbers into an array:

```python
self.tabwrite("myarray", [0.0, 0.5, 1.0])
```

Options:

- `resize=True` to resize the Pd array to fit the input length
- `redraw=False` to skip GUI redraw (useful for performance)

```python
self.tabwrite("myarray", [0.0, 0.5, 1.0], resize=True, redraw=True)
```

---

## Logging and errors

### `self.logpost(loglevel, *args, prefix=True)`

Posts to the Pd console. `loglevel` is an integer (Pd uses levels like error/log/debug depending on your build and settings).

```python
self.logpost(2, "processing", 123)
self.logpost(2, "no prefix", prefix=False)
```

### `self.error(*args)`

Prints an error tagged with the object name:

```python
self.error("Something went wrong")
```

---

## Helpers

- `self.get_current_dir()` → returns the current canvas directory (string)
- `self.reload()` → reloads the current `.pd_py` file and updates the running object

---

## Clocks (timers)

Create a clock:

```python
self.clock = pd.new_clock(self, self._tick)
```

Supported clock methods:

- `clock.delay(ms: float)` schedule the callback after `ms` milliseconds
- `clock.unset()` cancel any pending callback

Note: the C core also defines a `set(...)` method on the clock type, but `delay(...)` and `unset()` are the stable, expected API.

---

## Receivers (global symbol binding)

Bind to a Pd symbol so you can receive messages sent to it:

```python
self.r = pd.new_receiver(self, "mysym", self.received)

def received(self, x):
    self.logpost(2, "got", x)
```

The callback receives arguments using the same conversion rules as inlet methods (`bang`/`float`/`symbol`/`list`/other).

Implementation note: the current C core binds on creation and unbinds when the receiver object is destroyed; it does not currently expose explicit `bind()`/`unbind()` methods in Python.

---

## Audio (tilde) objects

If your class defines a callable `perform(...)`, `py4pd` registers it as a DSP object.
For DSP to actually run, your class must also define a callable `dsp(sr, blocksize, inchans)` method that returns a boolean.

### `dsp(sr, blocksize, inchans) -> bool`

- `sr`: sample rate (float)
- `blocksize`: block size (int)
- `inchans`: number of signal inlets (int)

Return `True` to enable DSP, `False` to disable (outputs will be zeroed).

### `perform(inputs) -> tuple`

- `inputs` is a tuple of `inchans` Python lists, each list with `blocksize` floats

Return:

- for 1 signal outlet: a tuple of `blocksize` floats
- for N signal outlets: a tuple of N tuples, each `blocksize` floats

Example (1-in / 1-out):

```python
import puredata as pd
import math


class pysine(pd.NewObject):
    name = "pysine~"

    def __init__(self, args):
        self.inlets = pd.SIGNAL
        self.outlets = pd.SIGNAL
        self.phase = 0.0

    def dsp(self, sr, blocksize, inchans):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        return True

    def perform(self, inputs):
        freq = inputs[0]
        out = []
        for f in freq:
            self.phase += (2.0 * math.pi * float(f)) / self.sr
            if self.phase > 2.0 * math.pi:
                self.phase -= 2.0 * math.pi
            out.append(math.sin(self.phase))
        return tuple(out)
```
