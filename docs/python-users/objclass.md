
## <h2 style="text-align:center"> **PureData Object Class** </h2>
 
 With version `0.9.0`, I add to `py4pd` a class called `new_object`. I new instance of this class must be created using `pd.new_object`. With `pd.new_object` you have complete access to the creation of PureData Objects using `C` or `C++`. Here one example of the creation of a new PureData object with 3 methods, 1 for PureData lists, 1 for PureData bangs, and one for the floats. 
 
``` python
 
newobj = pd.new_object("myobj")
newobj.addmethod_list(listmethod)
newobj.addmethod_bang(bangmethod)
newobj.addmethod_float(floatmethod)
newobj.add_object()

```
 

 
## <h2 style="text-align:center"> **Embbeded Module with `py4pd`** </h2>

--------------------
### <h3 style="text-align:center"> **Write PureData Objects** </h3>
--------------------

#### `pd.add_object` 

You can create your own objects with Python. For that, you define the Python Function and add it as an object using `#!python pd.add_object()`.

??? danger "Breaking Changes"
	I had change how `pd.add_object` work from version `0.6` to version `0.7`. Now, me use the function and the Pure Data object. Instead of use this, `pd.add_object("mysumObject", "NORMAL", "myNewPdObjects", "mysumObject")` we use this `pd.add_object(mysumObject, "mysumObject")`.

=== "Parameters"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `Python Function` | Function that will be executed by the object.  |
    | `arg2`   | `String` | String to create the object. |

=== "Keywords"

    | Keyword     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `objtype`   | `pd` | The type of the object: `pd.VIS`, `pd.AUDIO`, `pd.AUDIOIN`, or `pd.AUDIOOUT`. Hiding this option will create a normal object.  |
    | `figsize`   | `Tuple` | Sets the pixel size of the object. Example: `figsize=(400, 200)` creates an object with a width of 400 and height of 200. |
    | `pyout`    | `Boolean` | Determines whether the output will be in PureData data types or Python data types. If set to Python, you can not use the data before the convertion to PureData with `py2pd` object. |
    | `no_outlet`    | `Boolean` | Creates an object with no outlets if set to `True`. |
    | `num_aux_outlets`| `int` | Set the number of auxiliar outlets. If you use 4, it means that the object will have 5 inlets, 4 auxiliar and the main outlet (0). |
    | `added2pd_info` | `Boolean` | Prints the message `"[py4pd]: Object {objectName} added."` when set to `True`. |
    | `helppatch` | `String` | Personalized help patch, it always must be inside the `help` folder. |
    | `ignore_none_return` | `Boolean` | When `True` it ignores all things that return None. |
    | `objimage` | `String` | Set the standard image for `pd.VIS` objects. When you create the object it will load this image. |


=== "Examples"

    ``` python

    pd.add_object(myFunction, "mypyobj", 
                objtype=pd.VIS, figsize=(400, 200), 
                pyout=True, no_outlet=False, added2pd_info=False)

    ```

    ``` py

    import pd


    def mysumObject(a, b, c, d):
        return a + b + c + d

    def libraryname_setup():
        pd.add_object(mysumObject, "mysumObject")

        # My License, Name and University, others information
        pd.print("", show_prefix=False)
        pd.print("GPL3 2023, Your Name", show_prefix=False)
        pd.print("University of São Paulo", show_prefix=False)
        pd.print("", show_prefix=False)

    ```

    Here we add the function `mysumObject` in PureData enviroment. For more infos read the [Python Objects](https://www.charlesneimog.com/py4pd/researchers/) page.

    <p align="center">
        <img src="../../examples/createobj/mynewpdobject.png" width="50%" alt="My New PD Object">
    </p>



