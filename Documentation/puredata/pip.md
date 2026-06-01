## Package Management with `py.pip`

One of Python's best features is its massive library of packages. You can easily install them in Pure Data using the `[py.pip]` object.

**Availability:** The `[py.pip]` object becomes available automatically as soon as you load any `py4pd` object in your patch.

### How it Works

- **Safe for Pd:** It runs in the background, so the installation process won't freeze Pure Data or interrupt your audio.
- **Isolated Environment:** Packages are safely installed into a dedicated virtual environment created just for `py4pd`.

### Usage Example

To install packages, send a message starting with `pip install` followed by the libraries you need:

```text
[pip install librosa pytorch]
|
[py.pip]
```

