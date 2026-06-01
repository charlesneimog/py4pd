## DSP (Signal Processing)

To use Python to process `dsp`, you need to define two function in the object. `self.dsp` and `self.perform`. `self.dsp` must return `True` or `False`, for sucess or fail. And `self.perform` must return a `tuple` with the output output of the object.

### Signal declaration

**Example:**
```python
class SimpleGain(pd.NewObject):
    name = "gain"

    def __init__(self, args):
        self.inlets = (pd.SIGNAL, pd.DATA)   # inlet 0: audio, inlet 1: gain
        self.outlets = (pd.SIGNAL)            # outlet 0: audio
        self.gain = 1.0

    def in_1_float(self, f):
        self.gain = max(0.0, f)
```

---

### `dsp`

DSP initialization.

**Example:**
```python
class Oscillator(pd.NewObject):
    name = "oscillator~"

    def __init__(self, args):
        self.inlets = (pd.SIGNAL)
        self.outlets = (pd.SIGNAL)
        self.phase = 0.0

    # you can not change this signature
    def dsp(self, sr, block_size, audio_n_in):
        pd.post(f"DSP update: sr {sr} | block: {block_size} | in {audio_n_in}")
        return True

```

---

### `perform`

Signal processing callback.

**Example:**
```python
class SimpleGain(pd.NewObject):
    name = "simplegain"

    # ... __init__ as above ...

    # you cannot change this signature
    def perform(self, ins):
        in_sig = ins[0]
        g = self.gain
        return tuple(sample * g for sample in in_sig)
```
