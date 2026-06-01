To be able to process the input from inlets, you must define methods for each inlet using its index followed by uts selector. If you don't know what is selectors in Pd enviroment, check PureData Documentation [Chapter 2.4.1](https://msp.ucsd.edu/Pd_documentation/2.theory.of.operation.htm#s2.4.1){_target="blank"}.

### Inlet creation

You must define the numbers of inlets in the `self.__init__`, you must especify this number, even when it is 0.

``` python
def __init__(self, args):
    self.inlets = 4 # 4 inlets
    self.outlets = 1 # 1 outlet
    self.gain = 1.0
```

## Selectors

| Selector | Method signature |
|----------|------------------|
| `bang` | `in_N_bang(self)` |
| `float` | `in_N_float(self, f)` |
| `symbol` | `in_N_symbol(self, s)` |
| `list` | `in_N_list(self, xs)` |
| Custom | `in_N_custom(self, *args)` |
| `pyobj` | `in_N_pyobj(self, obj)` |


## Dynamic inlet creation

For some object, can be necessary to create methods in a dynamic way.

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

