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

