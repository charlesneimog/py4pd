### `out`

Send data to a Pd object outlet.

**Parameters:**

- `outlet`: 0‑based outlet index
- `type`: `pd.FLOAT`, `pd.SYMBOL`, `pd.LIST`, or `pd.PYOBJECT`
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
