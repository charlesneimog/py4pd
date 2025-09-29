<p align="center">
  <a href="https://charlesneimog.github.io/py4pd/">
    <img src="https://raw.githubusercontent.com/charlesneimog/py4pd/master/Documentation/assets/py4pd.svg" alt="Logo" width=100 height=58>
  </a>
  <h1 align="center">py4pd</h1>
  <h4 align="center">Python in the PureData environment.</h4>
</p>
<p align="center">
    <a href="https://github.com/charlesneimog/py4pd/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPL--v3-blue.svg" alt="License"></a>
    <a href="https://github.com/charlesneimog/py4pd/releases/latest"><img src="https://img.shields.io/github/release/charlesneimog/py4pd.svg?include_prereleases" alt="Release"></a>
    <a href="https://doi.org/10.5281/zenodo.10247117"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10247117.svg" alt="DOI"></a>
</p>
    
<p align="center">
  <a href="https://github.com/charlesneimog/py4pd/actions/workflows/Builder.yml"><img src="https://github.com/charlesneimog/py4pd/actions/workflows/Builder.yml/badge.svg?branch=master"></a>
</p>

py4pd allows write PureData Objects using Python instead of C/C++. The main goal is to allow easy IA, Scores, Graphics, and bring to Pd possibilities with array, list and others types. With Python, you can:
* Parse svg/draws as scores;
* OpenMusic functions in libraries like `om_py`, `music21`, `neoscore`, and others;
* Sound analisys with `magenta`, `librosa`, and `pyAudioAnalaysis`;

## Wiki | How to install and Use

* Go to [Docs](https://charlesneimog.github.io/py4pd) (outdated for now), check `Sources/py4pd` folder for examples.

### New Pd Object using Python

``` py
import puredata as pd
import os

class pymetro(pd.NewObject):
    name: str = "pymetro"

    def __init__(self, args):
        self.inlets = 2
        self.outlets = 1
        self.toggle = False
        if len(args) > 0:
            self.time = float(args[0])
        else:
            self.time = 1000
        self.metro = pd.new_clock(self, self.tick)
        self.args = args

    def in_2_float(self, f: float):
        self.time = f

    def in_1_float(self, f: float):
        if f:
            self.toggle = True
            self.tick()
        else:
            self.metro.unset()
            self.toggle = False

    def in_1_reload(self, args: list):
        self.reload()

    def tick(self):
        if self.toggle:
            self.metro.delay(self.time)
        self.out(0, pd.SYMBOL, "test238")
``` 

## Building from Source

* To build from the source code:
``` sh
cmake . -B build -DPYVERSION=3.12
cmake --build build
```

On windows you need Mingw64.


