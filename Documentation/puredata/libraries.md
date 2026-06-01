# Single-File Libraries

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
