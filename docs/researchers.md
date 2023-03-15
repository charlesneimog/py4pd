# Researchers

With the `py4pd` version `>` that `0.5.0-beta` it is possible to create new PureData objects using Python. For that, you need to declare your functions and then create a function called `py4pdLoadObjects`. See the Python Code:

``` py title="myNewPdObjects.py"

import pd


def mysumObject(a, b, c, d):
    return a + b + c + d


def py4pdLoadObjects():
    pd.addobject("mysumObject", "NORMAL", "myNewPdObjects", "mysumObject")

    # My License, Name and University, others information
    pd.print("", show_prefix=False)
    pd.print("GPL3 2023, Your Name", show_prefix=False)
    pd.print("University of São Paulo", show_prefix=False)
    pd.print("", show_prefix=False)

```

In the code above, we are creating a new object called `mysymObject`. It is saved inside an script called `myNewPdObjects.py`. To load this script in PureData how need to follow these steps:

* Copy the script `myNewPdObjects.py` for the `resources/scripts` inside `py4pd` folder or put it on side of your PureData patch.
* Create a new `py4pd` with this config: `py4pd -library myNewPdObjects`.
* Add the new object, in this case `mysumObject`.

Following this steps we have this patch:

<p align="center">
    <img src="../examples/createobj/mynewpdobject.png" width="50%"</img>
</p>
