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
    name = "py.receiver"

    def __init__(self, args):
        self.inlets = 0
        self.outlets = 1
        self.receiver = pd.new_receiver(self, "global_in", self._on_message)

    def _on_message(self, selector, *args):
        # selector is "float", "bang", "symbol", "list", or custom
        self.out(0, pd.SYMBOL, f"Got {selector}: {args}")
```
