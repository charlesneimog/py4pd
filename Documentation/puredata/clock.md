### `new_clock`

Create a Pd scheduler clock. Returns a `Clock` object.

**Parameters:**
- `owner`: The object instance (typically `self`)
- `callback`: Callable with signature `callback()`

**Example:**
```python
class DelayedPrint(pd.NewObject):
    name = "clock.example1"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 0
        self.clock = pd.new_clock(self, self._on_timer)

    def in_0_float(self, f):
        self.clock.delay(f)  # trigger after f ms

    def _on_timer(self):
        pd.post("Timer fired!")
```

---

### `delay` (clock method)

Schedule callback after relative milliseconds.

**Example:**
```python
class AutoBang(pd.NewObject):
    name = "clock.autobang"

    def __init__(self, args):
        self.inlets = 0
        self.outlets = 1
        self.clock = pd.new_clock(self, self._bang)
        self.clock.delay(1000)  # bang after 1 second

    def _bang(self):
        self.out(0, pd.FLOAT, 1)
        self.clock.delay(1000)  # repeat
```

---

### `set` (clock method)

Schedule callback at absolute system time.

**Example:**
```python
class Alarm(pd.NewObject):
    name = "clock.alarm"

    def __init__(self, args):
        self.inlets = 1
        self.outlets = 1
        self.clock = pd.new_clock(self, self._ring)

    def in_0_float(self, seconds_from_now):
        import time
        absolute_time = time.monotonic() + seconds_from_now
        self.clock.set(absolute_time)

    def _ring(self):
        self.out(0, pd.SYMBOL, "ALARM!")
```

---

### `unset` (clock method)

Cancel any pending schedule.

**Example:**
```python
class CancelableTimer(pd.NewObject):
    name = "clock.timer"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.clock = pd.new_clock(self, self._fire)

    def in_0_float(self, ms):
        self.clock.delay(ms)

    def in_1_bang(self):
        self.clock.unset()  # cancel pending
        self.out(0, pd.SYMBOL, "cancelled")
```

---
