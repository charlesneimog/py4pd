# Intro

Base class for all custom Pd objects.

## Instance Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `self.name` | str | class attribute | Class attribute name visible in Pd |
| `self.inlets` | int | 1 | Number of inlets |
| `self.outlets` | int | 1 | Number of outlets |

**Example:**
```python
class MyProcessor(pd.NewObject):
    # must be class attribute
    name = "proc"  

    def __init__(self, args):
        self.inlets = 3   # three inputs
        self.outlets = 2  # two outputs
```

---
