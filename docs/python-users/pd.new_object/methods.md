You can create your objects with Python. For the complex, you create a new class using `pd.new_object()` and define configurations, object types, etc using the methods of this class.

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



