With the `py4pd` it is possible to create new PureData objects using just Python. For that, you need to declare your Python functions and then create a function called `libraryname_setup` or `py4pdLoadObjects`. Inside this function, for simple objects you can use `pd.add_object`, for more complex objects, with lot of methods, you need to use the class returned from `pd.new_object()`.

Before understand the funcions, if you want to create libraries we highly suggest the folder organization showed below.

!!! note

    _The folder name must be the same name of the `.py` file. In this example, it must be called `libraryname`_,


!!! warning ""

    ```
    .
    ├── help-patches/ # helpPatches
        ├── myfunction1-help.pd
        └── myfunction2-help.pd
    ├── src/
        ├── setoffunctions1.py # you can organize/name this folder as you want, this is just one example.
        └── setoffunctions2.py
    ├── resources/
        ├── imagetosomeVISobject.png # for pd.VIS objects, you can set standard images of png or gif.
        └── imagetootherVISobject.gif
    ├── libraryname.py # here we have the libraryname_setup() or py4pdLoadObjects().
    ├── README.deken.pd # If you upload it on Deken, this will open immediately after installation.
    └── README.md # Ordinary readme for Github.

    ```

See the some libraries in:

- [orchidea](https://github.com/charlesneimog/orchidea)
- [py4pd-upic](https://github.com/charlesneimog/py4pd-upic)
- [py4pd-score](https://github.com/charlesneimog/py4pd-score)
- [py4pd-ji](https://github.com/charlesneimog/py4pd-ji)

---

### Basic Example

See the Python Code:

```py title="libraryname.py"

import pd
from src.setoffunctions1 import myfunction1
from src.setoffunctions2 import listmethod, floatmethod


def libraryname_setup():
    pd.add_object(myfunction1, "mysumObject")
    # simple objects, already it is possible to create mysumObject in Pd Patches

    # More complex objects
    myobj = pd.new_object("myobj") # create a class for the object
    myobj.addmethod_list(listmethod) # add method for lists
    myobj.addmethod_float(floatmethod) # add method for float
    myobj.add_object() # This call the function that make the object avaible in PureData patches.

```

<p align="center">
    Following this steps we have this patch:
    <img src="../../examples/createobj/mynewpdobject.png" width="50%"</img>
</p>
