### `get_current_dir`

Return the directory of the Pd canvas.

**Example:**
```python
class FileLoader(pd.NewObject):
    name = "fileloader"

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
    name = "arraybang"

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
    name = "validator"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_float(self, f):
        if f < 0:
            self.error("Negative value not allowed:", f)
            return
        self.out(0, pd.FLOAT, f)
```
