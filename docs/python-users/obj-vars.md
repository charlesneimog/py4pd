The following functions serve to instantiate variables for the object. The utilization of global variables is discouraged, primarily due to the potential for errors. For instance, consider the scenario where you create a sinusoidal object. In order to generate a continuous sinusoidal, it is crucial to store some pertinent values, as `phrase` for example. If the variables are saved globally and multiple objects are created, errors may arise. In the sinusoidal case, all objects would store the phrase in the same variable, rendering the sinusoidal wrong. In examples like this, it is appropriate to employ the objects `pd.set_obj_var`, `pd.get_obj_var` and `pd.accum_obj_var`.

---

## `pd.set_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    This function set a value for a object internal variable.
    
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

## `pd.accum_obj_var`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    This function will create a list with all things that you save in it. `py.collect` use this function and I believe that will be usefull just in loops contexts.
    
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

## `pd.get_obj_var`

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

## `pd.clear_obj_var`

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



