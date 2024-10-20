## Object Internal Variables

The following functions serve to instantiate variables for the object. The utilization of global variables is discouraged, primarily due to the potential for errors. For instance, consider the scenario where you create a sinusoidal object. To generate a continuous sinusoidal, it is crucial to store some pertinent values, such as `phrase` for example. If the variables are saved globally and multiple objects are created, errors may arise. In the sinusoidal case, all objects would store the phrase in the same variable, rendering the sinusoidal wrong. In examples like this, it is appropriate to employ the objects `pd.set_obj_var`, `pd.get_obj_var` and `pd.accum_obj_var`.

---

###`pd.set_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    This function sets a value for an object internal variable.
    
    ``` python
    import pd
	    
    def accum_thing(thing):
        pd.set_obj_var("my_obj_internal_var", thing)

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Variable Name. |
    | `arg2`   | `Python Object` | Python object to save |

</div>

###`pd.accum_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    This function will create a list with all the things that you save in it. `py.collect` uses this function and I believe that will be useful just in loop contexts.
    
    ``` python
    import pd
	    
    def accum_thing(thing):
        pd.accum_obj_var("my_obj_internal_var", thing)

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Variable Name. |
    | `arg2`   | `Python Object` | Python object to save |

</div>

###`pd.get_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It gets the value saved by `pd.set_obj_var` or `pd.accum_obj_var`.
    
    ``` python
    import pd
	    
    def get_thing(thing):
        pd.get_obj_var("my_obj_internal_var")

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Variable Name. |

</div>

###`pd.clear_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Set the variable to `None` and clear the memory.
    
    ``` python
    import pd
	    
    def get_thing(thing):
        pd.clear_obj_var("my_obj_internal_var")

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Variable Name. |

</div>

## Object Internal Variables

For musicians, it is important to put things in time. With the functions presented here, you can call specific functions using `onset` values. For example, in [py4pd-upic](https://github.com/charlesneimog/py4pd-upic) I use `pd.add_to_player` to play svg draws that are used to control parameters. So each `svg` elements trigger some specific function at a specific point in time.

!!! danger "Set the right configuration"
    <p style="font-size: 16px;">To use these `methods` the Python Object must be configurable with `playable` as `#!python True`. See [configuration](pd.new_object/config.md#playable) for the `player` object.</p>

### `pd.add_to_player`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns a unique string.
    
    ``` python
    import pd
	    
    def add_to_player(onset, thing):
        pd.add_to_player(onset, thing)

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `int` | Onset in milliseconds. |
    | `arg2`   | `Python Object` | Represents any Python entity; its output corresponds to the timestamp of onset. |

</div>

---

### `pd.clear_player`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Used to clear all events in the player.
    
    ``` python
    import pd
	    
    def clear_player():
        pd.clear_player()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no args for this function.

</div>

---





