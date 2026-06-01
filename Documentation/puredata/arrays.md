# Array Operations

## `tabread`

Read a Pd array. Returns tuple of floats.

**Example:**
```python
class ArrayReader(pd.NewObject):
    name = "arrayread"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1

    def in_0_symbol(self, array_name):
        data = self.tabread(array_name)
        self.out(0, pd.PYOBJECT, data)  # output as Python tuple
```

---

## `tabwrite`

Write numeric data to a Pd array.

**Example:**
```python
class ArrayWriter(pd.NewObject):
    name = "arraywrite"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 0

    def in_0_symbol(self, array_name):
        self.array_name = array_name

    def in_1_list(self, data):
        # Write without resize, skip redraw for speed
        self.tabwrite(self.array_name, data, resize=False, redraw=False)
```

