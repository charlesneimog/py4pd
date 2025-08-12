# `NewObject` Class

All `py4pd` object classes are created from the base class `puredata.NewObject`. 

```python
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro" # object name, must be exactly the same as the file name (pymetro.pd_py)

    def __init__(self, args):
        # Object initializer
        pass
```

