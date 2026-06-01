# Creating Python Pd Objects

## Module: `puredata`

This module provides the interface between Python and the Pure Data (Pd) engine.

```python
import puredata as pd
```

---

## Global Constants

| Constant | Type | Description |
|----------|------|-------------|
| `puredata.FLOAT` | int | Pd `A_FLOAT` – for float output |
| `puredata.SYMBOL` | int | Pd `A_SYMBOL` – for symbol output |
| `puredata.LIST` | int | Pd `A_GIMME` – for list output |
| `puredata.PYOBJECT` | int | Internal – passes Python object pointer |
| `puredata.SIGNAL` | int | For signal inlet/outlet declaration |
| `puredata.DATA` | int | For message inlet/outlet declaration |

---

## Global Functions

### `post`

Print a message to the Pd console. Not traceable to a specific object.

**Parameters:** `*args` – Variadic arguments, concatenated as strings.

**Example:**
```python
class MyObject(pd.NewObject):
    def __init__(self, args):
        pd.post("Object created with", len(args), "arguments")
```

---

### `hasgui`

Return `True` if Pd is running with a graphical user interface.

**Example:**
```python
class GuiAware(pd.NewObject):
    def __init__(self, args):
        if pd.hasgui():
            pd.post("GUI is available")
        else:
            pd.post("Running headless")
```

---

### `get_sample_rate`

Return the current audio sample rate in Hz.

**Example:**
```python
class Sampler(pd.NewObject):
    def __init__(self, args):
        self.sr = pd.get_sample_rate()
        pd.post("Sample rate:", self.sr)
```

---

## Clock Management

### `new_clock`

Create a Pd scheduler clock. Returns a `Clock` object.

**Parameters:**
- `owner`: The object instance (typically `self`)
- `callback`: Callable with signature `callback()`

**Example:**
```python
class DelayedPrint(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 0
        self.clock = pd.new_clock(self, self._on_timer)

    def in_0_float(self, f):
        self.clock.delay(f)  # trigger after f ms

    def _on_timer(self):
        pd.post("Timer fired!")
```

---

### `delay` (clock method)

Schedule callback after relative milliseconds.

**Example:**
```python
class AutoBang(pd.NewObject):
    def __init__(self, args):
        self.inlets = 0
        self.outlets = 1
        self.clock = pd.new_clock(self, self._bang)
        self.clock.delay(1000)  # bang after 1 second

    def _bang(self):
        self.out(0, pd.FLOAT, 1)
        self.clock.delay(1000)  # repeat
```

---

### `set` (clock method)

Schedule callback at absolute system time.

**Example:**
```python
class Alarm(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1
        self.clock = pd.new_clock(self, self._ring)

    def in_0_float(self, seconds_from_now):
        import time
        absolute_time = time.monotonic() + seconds_from_now
        self.clock.set(absolute_time)

    def _ring(self):
        self.out(0, pd.SYMBOL, "ALARM!")
```

---

### `unset` (clock method)

Cancel any pending schedule.

**Example:**
```python
class CancelableTimer(pd.NewObject):
    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.clock = pd.new_clock(self, self._fire)

    def in_0_float(self, ms):
        self.clock.delay(ms)

    def in_1_bang(self):
        self.clock.unset()  # cancel pending
        self.out(0, pd.SYMBOL, "cancelled")
```

---

## Receiver (Global Bindings)

### `new_receiver`

Bind to a global Pd receive symbol.

**Parameters:**
- `owner`: The object instance (typically `self`)
- `symbol`: Pd symbol name (string)
- `callback`: Callable with signature `callback(selector, *args)`

**Example:**
```python
class GlobalListener(pd.NewObject):
    def __init__(self, args):
        self.inlets = 0
        self.outlets = 1
        self.receiver = pd.new_receiver(self, "global_in", self._on_message)

    def _on_message(self, selector, *args):
        # selector is "float", "bang", "symbol", "list", or custom
        self.out(0, pd.SYMBOL, f"Got {selector}: {args}")
```

---

## Class: `puredata.NewObject`

Base class for all custom Pd objects.

### Instance Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `self.inlets` | int | 1 | Number of inlets |
| `self.outlets` | int | 1 | Number of outlets |
| `self.name` | str | (from class) | Object name visible in Pd |

**Example:**
```python
class MyProcessor(pd.NewObject):
    def __init__(self, args):
        self.inlets = 3   # three inputs
        self.outlets = 2  # two outputs
        self.name = "my.proc"  # custom Pd name
```

---

## Output Methods

### `out`

Send data to a Pd outlet.

**Parameters:**
- `outlet`: 0‑based outlet index
- `type`: `FLOAT`, `SYMBOL`, `LIST`, or `PYOBJECT`
- `value`: Data to output

**Example (converter):**
```python
class topd(pd.NewObject):
    name = "py.2pd"

    def __init__(self, _):
        self.inlets = 1
        self.outlets = 1

    def in_0_pyobj(self, args):
        if type(args) in (float, int):
            self.out(0, pd.FLOAT, args)
        elif type(args) == str:
            self.out(0, pd.SYMBOL, args)
        elif type(args) == list:
            self.out(0, pd.LIST, args)
```

**Example (PYOBJECT output):**
```python
class topy(pd.NewObject):
    name = "py.2py"

    def __init__(self, _):
        self.inlets = 1
        self.outlets = 1

    def in_0_list(self, args):
        self.out(0, pd.PYOBJECT, args)

    def in_0_float(self, f):
        self.out(0, pd.PYOBJECT, f)
```

---

## Array Operations

### `tabread`

Read a Pd array. Returns tuple of floats.

**Example:**
```python
class ArrayReader(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_symbol(self, array_name):
        data = self.tabread(array_name)
        self.out(0, pd.PYOBJECT, data)  # output as Python tuple
```

---

### `tabwrite`

Write numeric data to a Pd array.

**Example:**
```python
class ArrayWriter(pd.NewObject):
    def __init__(self, args):
        self.inlets = 2
        self.outlets = 0

    def in_0_symbol(self, array_name):
        self.array_name = array_name

    def in_1_list(self, data):
        # Write without resize, skip redraw for speed
        self.tabwrite(self.array_name, data, resize=False, redraw=False)
```

---

## Environment Methods

### `get_current_dir`

Return the directory of the Pd canvas.

**Example:**
```python
class FileLoader(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_symbol(self, filename):
        import os
        dirpath = self.get_current_dir()
        fullpath = os.path.join(dirpath, filename)
        with open(fullpath, 'r') as f:
            self.out(0, pd.SYMBOL, f.read())
```

---

### `reload`

Hot‑reload the source file.

**Example:**
```python
class DevObject(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 0

    def in_0_bang(self):
        if self.reload():
            pd.post("Reload successful")
        else:
            pd.post("Reload failed")
```

---

## Logging

### `logpost`

Thread‑safe logging to Pd console.

**Example:**
```python
class pyprint(pd.NewObject):
    name = "py.print"

    def __init__(self, _):
        self.inlets = 1
        self.outlets = 0

    def in_0_pyobj(self, args):
        self.logpost(2, args)  # loglevel 2 = debug
```

---

### `error`

Shorthand for error logging.

**Example:**
```python
class Validator(pd.NewObject):
    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_float(self, f):
        if f < 0:
            self.error("Negative value not allowed:", f)
            return
        self.out(0, pd.FLOAT, f)
```

---

## Inlet Callbacks

### Standard selectors

| Selector | Method signature |
|----------|------------------|
| `bang` | `in_N_bang(self)` |
| `float` | `in_N_float(self, f)` |
| `symbol` | `in_N_symbol(self, s)` |
| `list` | `in_N_list(self, xs)` |
| Custom | `in_N_custom(self, *args)` |
| `pyobj` | `in_N_pyobj(self, obj)` |

**Example (dynamic inlet creation):**
```python
class pyappend(pd.NewObject):
    name = "py.append"

    def __init__(self, args):
        self.inlets = int(args[0]) if args else 2
        self.outlets = 1
        self.dict = {}
        for i in range(1, self.inlets + 1):
            self._add_inlet(i)

    def _add_inlet(self, idx):
        def handler(args, idx=idx):
            self.dict[idx] = args if type(args) == list else [args]
            if idx == 1:
                output = []
                for i in range(1, self.inlets + 1):
                    output.extend(self.dict.get(i, []))
                self.out(0, pd.PYOBJECT, output)

        for sel in ['list', 'float', 'symbol', 'pyobj']:
            setattr(self, f"in_{idx}_{sel}", handler)
```

---

## DSP (Signal Processing)

### Signal declaration

**Example:**
```python
class SimpleGain(pd.NewObject):
    def __init__(self, args):
        self.inlets = (pd.SIGNAL, pd.DATA)   # inlet 0: audio, inlet 1: gain
        self.outlets = (pd.SIGNAL)            # outlet 0: audio
        self.gain = 1.0

    def in_1_float(self, f):
        self.gain = max(0.0, f)
```

---

### `perform`

Signal processing callback.

**Example:**
```python
class SimpleGain(pd.NewObject):
    # ... __init__ as above ...

    # you cannot change this signature
    def perform(self, ins):
        in_sig = ins[0]
        g = self.gain
        return tuple(sample * g for sample in in_sig)
```

---

### `dsp` (optional)

Override for DSP initialization.

**Example:**
```python
class Oscillator(pd.NewObject):
    def __init__(self, args):
        self.inlets = (pd.SIGNAL)
        self.outlets = (pd.SIGNAL)
        self.phase = 0.0

    # you can not change this signature
    def dsp(self, sr, block_size, audio_n_in):
        pd.post(f"DSP update: sr {sr} | block: {block_size} | in {audio_n_in}")
        return True

```

---

## Package Management with `py.pip`

One of Python's best features is its massive library of packages. You can easily install them in Pure Data using the `[py.pip]` object.

**Availability:** The `[py.pip]` object becomes available automatically as soon as you load any `py4pd` object in your patch.

### How it Works

- **Safe for Pd:** It runs in the background, so the installation process won't freeze Pure Data or interrupt your audio.
- **Isolated Environment:** Packages are safely installed into a dedicated virtual environment created just for `py4pd`.

### Usage Example

To install packages, send a message starting with `pip install` followed by the libraries you need:

```text
[pip install librosa pytorch]
|
[py.pip]
```

---

## Single-File Libraries

For simpler projects or when you need something that is easy to share, you can bundle multiple objects into a single file to act as a library.

### Loading the Library

To access all objects contained within a library file, you must explicitly load the library using the `declare` object:

```text
[declare -lib mylib]
```

!!! danger "Loading the library using the library name will not work!"

After this declaration, all classes defined in `mylib.pd_py` become available as Pd objects.

### Library File Structure

Create a file named `mylib.pd_py` and define several distinct classes within it:

```python title="mylib.pd_py"
import puredata as pd

class class_myobj1(pd.NewObject):
    name = "myobj1"

    def __init__(self, args):
        self.inlets = 0
        self.outlets = 0
        pd.post("Object 1 created")

class class_myobj2(pd.NewObject):
    name = "myobj2"

    def __init__(self, args):
        self.inlets = 0
        self.outlets = 0
        pd.post("Object 2 created")
```

**Important:** The class name (e.g., `class_myobj1`) is what you type in the Pd patch, unless you override the `name` attribute. In the example above, the `name` attribute sets the Pd object name to `myobj1` and `myobj2`.

### Real-World Library Example

```python title="pyutils.pd_py"
import puredata as pd

class pyappend(pd.NewObject):
    """Concatenate all inputs into one list."""
    name = "py.append"

    def __init__(self, args):
        self.inlets = int(args[0]) if args else 2
        self.outlets = 1
        self.dict = {}
        for i in range(1, self.inlets + 1):
            self._add_inlet(i)

    def _add_inlet(self, idx):
        def handler(args, idx=idx):
            self.dict[idx] = args if type(args) == list else [args]
            if idx == 1:
                output = []
                for i in range(1, self.inlets + 1):
                    output.extend(self.dict.get(i, []))
                self.out(0, pd.PYOBJECT, output)
        for sel in ['list', 'float', 'symbol', 'pyobj']:
            setattr(self, f"in_{idx}_{sel}", handler)


class pyfirst(pd.NewObject):
    """Extract first element of a list."""
    name = "py.first"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_pyobj(self, args):
        try:
            self.out(0, pd.PYOBJECT, args[0])
        except Exception:
            self.out(0, pd.PYOBJECT, None)


class pylength(pd.NewObject):
    """Return length of a list as float."""
    name = "py.length"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_pyobj(self, args):
        self.out(0, pd.FLOAT, len(args))

    def in_0_list(self, args):
        self.out(0, pd.FLOAT, len(args))
```

### Using the Library in a Patch

```text
# Load the library
[declare -lib pyutils]

# Now use the objects
[py.append 3]
[py.first]
[py.length]
```

---

## Complete Working Example

```python
import puredata as pd


class Gain(pd.NewObject):
    name = "gain~"

    def __init__(self, args):
        self.inlets = (pd.SIGNAL, pd.DATA)
        self.outlets = pd.SIGNAL
        self.gain = float(args[0]) if args else 1.0

    def in_1_float(self, f):
        self.gain = max(0.0, f)

    def in_1_list(self, args):
        if args:
            self.gain = max(0.0, float(args[0]))

    def dsp(self, sr, block_size, audio_n_in):
        pd.post("dsp called")
        pd.post(f"DSP update: sr {sr} | block: {block_size} | in {audio_n_in}")
        return True

    def perform(self, ins):
        in_sig = ins[0]
        g = self.gain
        return tuple(sample * g for sample in in_sig)
```

---

## Version and Compatibility

This reference applies to `py4pd` version 1.2.3. Behavior not specified here is undefined and may change.
