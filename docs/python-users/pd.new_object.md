You can create your objects with Python. For the complex, you create a new class using `pd.new_object()` and define configurations, object types, etc using the methods of this class.



## Adding Methods for the Object

### `addmethod_bang`

This function will be executed when the object receives a bang.


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd
    import random

    def randomNumber():
        return random.randint(0, 100)


    def py4pdLoadObjects():
        random = pd.new_object("py.random")
        random.addmethod_bang(randomNumber)
        random.add_object()

    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/bang.gif" width="60%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>
    
</div>

### `addmethod_float`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd
    import random

    def randomNumber(limit):
        return random.randint(0, limit) 

    def py4pdLoadObjects():
        random = pd.new_object("py.floatRandom")
        random.addmethod_float(randomNumber)
        random.add_object()

    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/float.gif" width="60%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>
    
</div>


### `addmethod_symbol`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd

    def splitSymbol(symbol):
        # return a list with all chars separated
        return list(symbol)

    def py4pdLoadObjects():
        random = pd.new_object("py.splitSymbol")
        random.addmethod_symbol(splitSymbol)
        random.add_object()

    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/symbol.gif" width="90%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>
    
</div>

### `addmethod_list`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd

    def listmethod(list):
        newlist = [] 
        for x in list:
            newlist.append(x * x)
        return newlist

    def py4pdLoadObjects():
        random = pd.new_object("py.listmultiplier")
        random.addmethod_list(listmethod)
        random.add_object()

    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/list.gif" width="90%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>
    
</div>

### `addmethod_anything`


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd

    def anythingmethod(mything):
        return f"Received {mything}"

    def py4pdLoadObjects():
        random = pd.new_object("py.anything")
        random.addmethod_anything(anythingmethod)
        random.add_object()

    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/anything.gif" width="90%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>
    
</div>


### `addmethod`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Python Example__

    ```python
    import pd

    def portugues(a, b, c):
        return "Portuguese method"

    def english(d, e):
        return "English method"

    def german(f):
        return "German method"
        
    def dutch():
        return "Dutch method"    
        
    def py4pdLoadObjects():
        random = pd.new_object("py.methods")
        random.addmethod("casa", portugues)
        random.addmethod("home", english)
        random.addmethod("haus", german)
        random.addmethod("huis", dutch)
        random.add_object()
    ```

-   :fontawesome-brands-python: __PureData Example__

    <p align="center">
        <img src="../../../../examples/new_object/addmethod.gif" width="90%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);" alt="Scores">
    </p>

    !!! info "Inlets"
        
        Note that as the function with more arguments is portugues (with 3) arguments, the object will have 3 inlets.
    
</div>


### `add_object`

Don't be fooled by false class. `py4pd` still uses procedural language. After configuring your object, you need to use the `pd.add_object()` function for the object to be available in the PureData patch.

```python

import pd
import random

def randomNumber():
    return random.randint(0, 100)


def py4pdLoadObjects():
    myrandom = pd.new_object("py.random")
    myrandom.addmethod_bang(randomNumber)
    myrandom.add_object() # without this, py.random will not be available in the patch.
```

!!! danger

    <p>
    Without `random.add_object()` py.random will not be available in the patch.
    </p>



## Configuration

Some configuration for the object.

---
### Types

<div class="grid cards" markdown>


-   #### :fontawesome-brands-python: __`type`__

    !!! note ""

        Set the type of the object: `pd.NORMAL`, `pd.VIS`, `pd.AUDIOIN`, `pd.AUDIOOUT`, `pd.AUDIO`.

    ``` py
    def py4pdLoadObjects():
        myrandom = pd.new_object("py.random")
        myrandom.addmethod_bang(randomNumber)
        myrandom.type = pd.AUDIOIN
        myrandom.add_object() 

    ```

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `bool` | `pd.NORMAL` or `pd.VIS` or `pd.AUDIOIN` or `pd.AUDIOOUT` or `pd.AUDIO` |

</div>

---
### Outlets

<div class="grid cards" markdown>

-   #### :fontawesome-brands-python: __`require_n_of_outlets`__

    !!! note ""
        When set `True`, the creation of the object will require the user to set the number of outlets of the object.

    ``` py
    def py4pdLoadObjects():
        gateobj = pd.new_object("py.gate")
        gateobj.require_outlet_n = True
        gateobj.add_object() 

    ```

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `bool` | `True` to require to set the number of outlets. | 
    

-   #### :fontawesome-brands-python: __`py_out`__

    !!! note ""
        When set to `True`, the object will output `PyObject` instead of PureData data types. 

    ``` py
    def py4pdLoadObjects():
        gateobj = pd.new_object("py.gate")
        gateobj.py_out = True
        gateobj.add_object() 

    ```

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `bool` | `True` to require to set the number of outlets. | 


-   #### :fontawesome-brands-python: __`no_outlet`__

    !!! note ""
        When set to `True`, the object will be created without any outlet.

    ``` py
    def py4pdLoadObjects():
        obj = pd.new_object("myconfigObj")
        obj.no_outlet = True
        obj.add_object() 

    ```

-   #### :fontawesome-brands-python: __`n_extra_outlets`__

    !!! note ""
        When set to `True`, the object will be created without any outlet.

    ``` py
    def py4pdLoadObjects():
        obj = pd.new_object("py.if")
        obj.n_extra_outlets = 1 
        # the object will have 2 outlets.
        obj.add_object() 

    ```
</div>

---
### Output

<div class="grid cards" markdown>

-   #### :fontawesome-brands-python: __`py_out`__

    !!! note ""
        When set to `True`, the object will output `PyObject` instead of PureData data types. 

    ``` py
    def py4pdLoadObjects():
        gateobj = pd.new_object("py.gate")
        gateobj.py_out = True
        gateobj.add_object() 

    ```

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `bool` | `True` to require to set the number of outlets. | 


-   #### :fontawesome-brands-python: __`ignore_none`__

    !!! note ""
        When set to `True`, the object will not output anything if PyObject is `None`.

    ``` py
    def py4pdLoadObjects():
        obj = pd.new_object("myconfigObj")
        obj.ignore_none = True
        obj.add_object()

    ```

</div>

---
### Images

<div class="grid cards" markdown>


-   #### :fontawesome-brands-python: __`image`__

    !!! note ""

        Set the pathname for the default image for a `pd.VIS` object.

    ``` py
    def py4pdLoadObjects():
        patchZoom = pd.get_patch_zoom()
        if patchZoom == 1:
            scoreImage = "./resources/nozoom.gif"
        elif patchZoom == 2:
            scoreImage = "./resources/zoom.gif"
        
        # py.chord
        chordObj = pd.new_object("py.chord")
        chordObj.addmethod_anything(chord)
        chordObj.image = scoreImage
        chordObj.fig_size = (250, 250)
        chordObj.type = pd.VIS
        chordObj.add_object()

    ```

-   #### :fontawesome-brands-python: __`fig_size`__

    !!! note ""

        Set the height and width (respectivaly) of the image.

    ``` py
    def py4pdLoadObjects():
        patchZoom = pd.get_patch_zoom()
        if patchZoom == 1:
            scoreImage = "./resources/nozoom.gif"
        elif patchZoom == 2:
            scoreImage = "./resources/zoom.gif"
        
        # py.chord
        chordObj = pd.new_object("py.chord")
        chordObj.addmethod_anything(chord)
        chordObj.image = scoreImage
        chordObj.fig_size = (250, 250)
        chordObj.type = pd.VIS
        chordObj.add_object()


    ```

</div>

---
### Player


<div class="grid cards" markdown>

-   #### :fontawesome-brands-python: __`playable`__

    !!! note ""

        When set to `True`, it adds 3 methods for the object, `play`, `stop`, `clear`, this allows to use the function `pd.add_to_player`.

    ``` py

    def addtoplayer(onset, thing):
        pd.add_to_player(onset, thing)


    def py4pd_upic_setup():
        upic = pd.new_object("readsvg")
        upic.addmethod_anythin(readSvg)
        upic.playable = True
        upic.py_out = True
        upic.ignore_none_return = True
        upic.add_object()

    ```



</div>

