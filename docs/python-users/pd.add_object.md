You can create your objects with Python. For the simplest, you define the Python Function and add it as an object using `#!python pd.add_object()`.

It is crucial to emphasize that when an object uses `pd.add_object()` it is added as an anything object. It implies that if your object functions with individual floats rather than lists, `py4pd` does not provide additional 'protection' for types. In such cases, you are responsible for managing these different types on your own. For more complex objects it is recommended to utilize `pd.new_object()`.


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
    ``` py
    import pd
    from src.thing import myFunction 
	    
    def Py4pdLoadObjects():
        pd.add_object(myFunction, "mypyobj", 
                objtype=pd.VIS, figsize=(400, 200), 
                pyout=True, no_outlet=False, added2pd_info=False)

    ```

    
</div>

<div class="grid cards" markdown>

-   :fontawesome-solid-gear: __Arguments__
    
    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `Python Function` | Function that will be executed by the object.  |
    | `arg2`   | `String` | String to create the object. |
    | `**objtype`   | `pd` | The type of the object: `pd.VIS`, `pd.AUDIO`, `pd.AUDIOIN`, or `pd.AUDIOOUT`. Hiding this option will create a normal object.  |
    | `**figsize`   | `Tuple` | Sets the pixel size of the object. Example: `figsize=(400, 200)` creates an object with a width of 400 and height of 200. |
    | `**pyout`    | `Boolean` | Determines whether the output will be in PureData data types or Python data types. If set to Python, you can not use the data before the convertion to PureData with `py2pd` object. |
    | `**no_outlet`    | `Boolean` | Creates an object with no outlets if set to `True`. |
    | `**num_aux_outlets`| `int` | Set the number of auxiliar outlets. If you use 4, it means that the object will have 5 inlets, 4 auxiliar and the main outlet (0). |
    | `**added2pd_info` | `Boolean` | Prints the message `"[py4pd]: Object {objectName} added."` when set to `True`. |
    | `**helppatch` | `String` | Personalized help patch, it always must be inside the `help` folder. |
    | `**ignore_none_return` | `Boolean` | When `True` it ignores all things that return None. |
    | `**objimage` | `String` | Set the standard image for `pd.VIS` objects. When you create the object it will load this image. |
    
</div>



