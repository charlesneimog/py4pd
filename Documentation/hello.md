# Hello World

## Creating a `py4pd` Object

To define a new `py4pd` object, create a subclass of `puredata.NewObject`, name the object, and finally name the script with the name + `.pd_py`. For example, `pymetro.pd_py` for the object `pymetro`. Place the file in a place where Pd can find it. 

!!! Warning "Don't forget to load `py4pd` first"


### Example

```py
import puredata as pd

class pymetro(pd.NewObject):
    name: str = "pymetro"  # Name of the Pure Data object

    def __init__(self, args):
        self.inlets = 2    # Number of inlets
        self.outlets = 1   # Number of outlets
```

### Key Points

The Python class name (e.g., `pymetro`) can be any valid class name. The name attribute determines the name of the object inside Pure Data. `self.inlets` and `self.outlets` define the number of inlets and outlets for the object. When loading this object in Pure Data, use the name attribute value (`pymetro` in this example) as the object name.


## Input and Output

### Input 

The input design is inspired by the mature `pd-lua` project. For methods, use the format `in_<inlet_number>_<method>`. For example, to execute code when a `float` is received on inlet 1, define a method called `in_1_float`. Pd provides predefined methods that do not require a custom selector: `bang`, `float`, `symbol`, `list`, and `anything`. You can also create custom selectors (prefixes); for instance, `in_1_mymethod` will be executed when the message `mymethod` is sent to inlet 1 of the object.

### Output

To produce output, use the method `self.out`. For example, `self.out(0, pd.SYMBOL, "test238")` sends the symbol `"test238"` to outlet 0. The second argument specifies the data type, which can be `pd.SYMBOL` or `pd.FLOAT`. To output a list, use `pd.LIST` instead. To output `numpy.ndarray`, `class` and others you must use `pd.PYOBJECT`.

`py4pd` also implements the `PyObject` message, which allows you to share Python data types between `py4pd` objects. This enables the transfer of class instances, NumPy arrays, and other Python objects that are not supported by Pure Data’s traditional data types.

## Metronome Example

``` python
import puredata as pd


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

    def in_1_float(self, f: float):
        self.time = f

    def in_0_float(self, f: float):
        if f:
            self.toggle = True
            self.tick()
        else:
            self.metro.unset()
            self.toggle = False

    def tick(self):
        if self.toggle:
            self.metro.delay(self.time)
        self.out(0, pd.SYMBOL, "test238")
```

## Oscillator Example

This is a simple oscillator example

```python
import puredata as pd
import math


class pytest_tilde(pd.NewObject):
    name: str = "pytest~"

    def __init__(self, args):
        self.inlets = pd.SIGNAL
        self.outlets = pd.SIGNAL
        self.phase = 0

    def perform(self, input):
        blocksize = self.blocksize
        samplerate = self.samplerate

        out_buffer = []
        for i in range(blocksize):
            phase_increment = 2 * math.pi * input[0][i] / samplerate
            sample = math.sin(self.phase)
            out_buffer.append(sample)
            self.phase += phase_increment
            if self.phase > 2 * math.pi:
                self.phase -= 2 * math.pi
        out_tuple = tuple(out_buffer)
        return out_tuple

    def dsp(self, sr, blocksize, inchans):
        self.samplerate = sr
        self.blocksize = blocksize
        self.inchans = inchans
        return True
```


## Train AI Example


This is a complex and complet example to train AI using objects from `timbreLIBId`. This use `threading` to avoid block the Pd audio thread.

```python
import puredata as pd

import os
import random
import threading
import numpy as np
import librosa

from sklearn.metrics import classification_report
from catboost import CatBoostClassifier


class pytrain(pd.NewObject):
    name = "py.train"

    def __init__(self, args):
        # pd
        self.inlets = 2
        self.outlets = 2
        self.tabname = "train"
        self.redraw_tab_after = 120

        # train parameters
        self.max_offset = 256
        self.n_windows = 10
        self.iterations = 150
        self.fn_estimators = 150
        self.random_state = 42
        self.test_fraction = 0.2

        # folders
        self.trainfolder = ""
        self.folders = {}
        self.currtraindata = []
        self.currtestdata = []

        # datasets
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []

        # model
        self.clf = self._init_model()

    # ----------------------------
    # Model
    # ----------------------------
    def _init_model(self):
        # TODO: Implement more models
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=self.random_state,
            verbose=100,
            early_stopping_rounds=20,
        )

    # ----------------------------
    # Folder / Dataset Management
    # ----------------------------
    def _resolve_trainfolder(self, path):
        if os.path.exists(path):
            return path
        candidate = os.path.join(self.get_current_dir(), path)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"{path} folder not found")
        return candidate

    def in_1_trainfolder(self, args):
        self.trainfolder = self._resolve_trainfolder(args[0])
        self.folders = {
            f: os.path.join(self.trainfolder, f)
            for f in os.listdir(self.trainfolder)
            if os.path.isdir(os.path.join(self.trainfolder, f))
        }
        self.logpost(2, "Train folder: " + self.trainfolder)

    # ----------------------------
    # Audio processing
    # ----------------------------
    def _load_audio(self, filepath):
        sr = pd.get_sample_rate()
        y, _ = librosa.load(filepath, sr=sr)
        return y

    def _split_train_test(self, files):
        n_test = max(1, int(len(files) * self.test_fraction))
        test_files = random.sample(files, n_test)
        train_files = [f for f in files if f not in test_files]
        return train_files, test_files

    def _generate_variants(self, y, sr, mode="traindata"):
        """Retorna uma lista de versões do áudio (original + augmentadas se treino)"""
        variants = [y]  # sempre inclui o original

        if mode == "traindata":
            # Exemplo: time stretch
            value = random.uniform(0.7, 1.2)
            variants.append(librosa.effects.time_stretch(y=y, rate=value))
            value = random.uniform(0.7, 1.2)
            variants.append(librosa.effects.time_stretch(y=y, rate=value))

            # Exemplo: pitch shift
            value = random.uniform(-2, 2)
            variants.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=value))
            value = random.uniform(-2, 2)
            variants.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=value))

            # Exemplo: adicionar ruído
            value = random.uniform(0.005, 0.009)
            noise = np.random.normal(0, value, len(y))
            variants.append(y + noise)

        return variants

    def _process_file(self, filepath, label, target_list, mode):
        y = self._load_audio(filepath)
        sr = pd.get_sample_rate()
        variants = self._generate_variants(y, sr, mode=mode)

        for signal in variants:
            self._write_tab(signal)

            idx = random.randint(0, 1024)
            while True:
                self.out(1, pd.LIST, [idx, mode])
                if idx >= len(signal) - 2048:
                    break

                if mode == "testdata":
                    assert len(self.currtestdata) > 0
                    target_list.append((self.currtestdata, label))
                    self.currtestdata = []
                else:
                    assert len(self.currtraindata) > 0
                    target_list.append((self.currtraindata, label))
                    self.currtraindata = []

                idx += random.randint(512, 1024)

    def _write_tab(self, y):
        self.redraw += 1

        self.tabwrite(
            "train",
            y.tolist(),
            resize=True,
            redraw=(self.redraw % self.redraw_tab_after == 0),
        )

    # ----------------------------
    # Dataset Build
    # ----------------------------
    def get_train_mir(self):
        self.redraw = 0
        train_data, test_data = [], []

        for label, folder in self.folders.items():
            self.logpost(2, f"Processing {label}")
            all_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith((".aif", ".aiff", ".wav"))
            ]
            train_files, test_files = self._split_train_test(all_files)

            for f in test_files:
                self._process_file(f, label, test_data, "testdata")

            for f in train_files:
                self._process_file(f, label, train_data, "traindata")

        self.x_train, self.y_train = zip(*train_data) if train_data else ([], [])
        self.x_test, self.y_test = zip(*test_data) if test_data else ([], [])

        self.x_np_train = np.array(self.x_train)
        self.y_np_train = np.array(self.y_train)
        self.x_np_test = np.array(self.x_test)
        self.y_np_test = np.array(self.y_test)

        self.logpost(2, "Done!")

    # ----------------------------
    # Training / Export
    # ----------------------------
    def _train(self):
        self.clf.fit(
            self.x_np_train,
            self.y_np_train,
            eval_set=(self.x_np_test, self.y_np_test),
        )
        y_pred = self.clf.predict(self.x_np_test)
        self.logpost(2, classification_report(self.y_np_test, y_pred), prefix=False)
        self.logpost(2, "", prefix=False)
        self.logpost(2, "Training finished!")

    def in_0_train(self, args):
        self.logpost(2, "Training, wait...")
        t = threading.Thread(target=self._train, daemon=True)
        t.start()

    def in_0_export(self, args):
        file = args[0]
        path = os.path.join(self.get_current_dir(), file)
        if os.path.exists(path):
            self.logpost(1, "Model already exists, will replace it!")
        self.clf.save_model(path, format="onnx")
        self.logpost(2, f"Model exported to {path}")

    # ----------------------------
    # Utils
    # ----------------------------
    def in_0_printdata(self, _):
        self.logpost(2, f"Train samples: {len(self.x_train)}")
        self.logpost(2, f"Train labels: {len(self.y_train)}")
        assert len(self.x_train) == len(self.y_train)

        self.logpost(2, f"Test samples: {len(self.x_test)}")
        self.logpost(2, f"Test labels: {len(self.y_test)}")
        assert len(self.x_test) == len(self.y_test)

    def in_1_testdata(self, data):
        self.currtestdata = data

    def in_1_traindata(self, data):
        self.currtraindata = data

    def in_0_analyze(self, _):
        t = threading.Thread(target=self.get_train_mir, daemon=True)
        t.start()

    def in_0_randomload(self, args):
        all_files = [
            os.path.join(folder, f)
            for label, folder in self.folders.items()
            for f in os.listdir(folder)
            if f.endswith((".aif", ".aiff"))
        ]
        if not all_files:
            return None
        random_file = random.choice(all_files)
        y = self._load_audio(random_file)
        self.tabwrite("train", y.tolist(), resize=True)
```

For `patch` where I use this object and more info check [pd-onnx](https://github.com/charlesneimog/pd-onnx). Inside the `resources` folder also this youtube [example](https://www.youtube.com/watch?v=qIV0LigMuzo).
