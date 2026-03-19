# `puredata` Python API of `py4pd`

This page is a **reference** for what the C core exports in the `puredata` module and what methods exist on `puredata.NewObject`.

If you are writing `.pd_py` objects, you typically:

- `import puredata as pd`
- subclass `pd.NewObject`
- implement inlet methods like `in_1_float(...)`

---

## Module: `puredata`

### Constants

These are added during module init.

## Output type constants

- `puredata.FLOAT` (Pd `A_FLOAT`)
- `puredata.SYMBOL` (Pd `A_SYMBOL`)
- `puredata.LIST` (Pd `A_GIMME`)
- `puredata.PYOBJECT` (output Python Objects for Pd)

## Inlets types constants (for `self.inlets` / `self.outlets`)

- `puredata.SIGNAL`, relative to `&s_signal`;
- `puredata.DATA`, relative to `&s_anything`;

---

### `puredata.post(*args) -> bool`

Posts a message to the Pd console (not rastreable), 

---

### `puredata.hasgui() -> bool`

Returns non-zero when Pd is running with a GUI.

---

### `puredata.get_sample_rate() -> int`

Returns Pd sample rate.

---

### `puredata._print(*args, **kwargs) -> None`

Internal print used by the runtime (prints with Pd `logpost` level 2).

---

## `puredata.new_clock(owner, callback) -> puredata.clock`

Creates a Pd clock owned by `owner` (`self`). When it fires it calls `callback()`.

### `puredata.clock.delay(ms: float) -> bool`

Schedule `callback()` after `ms` milliseconds.

### `puredata.clock.unset() -> bool`

Cancel any scheduled callback.

### `puredata.clock.set(time: float) -> bool`

Schedule clock for an absolute systime. `True` on sucess.

---

### `puredata.new_receiver(owner, symbol: str, callback) -> puredata.receiver`

Binds to `symbol` (global Pd symbol) and forwards incoming messages to `callback`.

Notes based on the current C implementation:

- Binding happens immediately on creation and is automatically undone on destruction.
- The receiver object is primarily for lifetime management; treat it as an **opaque handle**.

---

## Class: `puredata.NewObject`

Your `.pd_py` object class must inherit from `puredata.NewObject`.

### Constructor

`puredata.NewObject(name: str)`

The C core uses this to ensure the module-level registry exists. In practice, you don’t call it directly; you inherit and let py4pd instantiate your class.

---

### Methods (available on your object instance)

#### `self.out(outlet: int, type: int, value) -> bool`

Send data to a Pd outlet.

- `type` must be one of: `puredata.FLOAT`, `puredata.SYMBOL`, `puredata.LIST`, `puredata.PYOBJECT`
- Outlet is 0-based.

Behavior:

- `FLOAT`: converts `value` to float and outputs a Pd float
- `SYMBOL`: converts `value` to string and outputs a Pd symbol
- `LIST`: requires a Python `list` of numbers/strings
- `PYOBJECT`: sends an internal pointer token so another py4pd object can receive the original Python object via `in_N_pyobj`

---

#### `self.tabread(name: str) -> tuple[float, ...]`

Read a Pd array (`garray`) into a Python tuple.

---

#### `self.tabwrite(name: str, data: list|tuple, resize=False, redraw=True) -> bool`

Write Python numeric data into a Pd array.

- `resize=True` will resize the Pd array to match input length
- `redraw=False` skips GUI redraw (useful for performance)

---

#### `self.get_current_dir() -> str | None`

Returns the directory of the current canvas.

---

#### `self.reload() -> bool`

Reloads the current `.pd_py` source file and swaps the running class instance.

Also updates:

- clock callbacks (rebinds by function name)
- DSP callback (`perform`) if DSP is active

---

#### `self.logpost(loglevel: int, *args, prefix=True) -> bool`

Posts a log message to the Pd console.

- If called from a non-main thread, the message is queued to the Pd main thread.
- `prefix=True` prints `[{object_name}]: ...`.

---

#### `self.error(*args) -> bool`

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
