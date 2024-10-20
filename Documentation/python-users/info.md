## Files and Folders


### `pd.get_patch_dir`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Returns pathname of the current patch folder.
    
    ``` py
    import pd
	    
    def getPatchDir():
        return pd.get_patch_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>


-------------------------------------- 


### `pd.get_temp_dir`



<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    `pd.get_temp_dir` returns a pathname for a temp-folder, **all files inside this folder are deleted when the PureData patch is closed** or when all the `py4pd` objects are deleted.

    ``` py
    import pd
	    
    def getTempDir():
        return pd.get_temp_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>

---

### `pd.get_py4pd_dir`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Returns the folder where the binary of `py4pd` is located.
    
    ``` py
    import pd
	    
    def getPy4pdDir():
        return pd.get_py4pd_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>


### `pd.get_pd_search_paths`



<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    `pd.get_pd_search_paths` returns all the folders in the PureData search path.
    
    ``` py
    import pd
	    
    def getPdSearchPaths():
        return pd.get_pd_search_paths()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>


## Messages and Debug

There are three messages used to print info in the PureData console, `pd.print`, `pd.logpost` and `pd.error`.

------------------
### `pd.print`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__


    The ordinary function `#!python print()` will not work in `py4pd` (unless you open PureData from the terminal). So if you want to debug or print some info from the PureData console you need to use `#!python pd.print`. 

    
    ``` py
    import pd
	    
    def printInfo():
        pd.print("ok") 
        # It prints "[Python] ok"
        pd.print("ok", show_prefix=False) 
        # It prints "ok".

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `Python Object` | Thing to print |

    ---
    | Kwargs | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `show_prefix`   | `Python Object` | When `False` don't print prefix |

    !!! info
        For Object written in Python, the prefix will be the Object Name.
    
</div>

------------------
### `pd.logpost`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    This function uses `logpost` in C PureData API to log messages using levels. For example, if you use `#!python logpost(4, "PureData message in level 4")`, the message will appear in the console just if the user had selected to show the messages of level 4 in PureData Console.
    
    ``` py
    import pd
	    
    def printInfoLevel():
        pd.logpost(1, "Level 1") 
        pd.logpost(2, "Level 2") 
        pd.logpost(3, "Level 3") 
        pd.logpost(4, "Level 4") 

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `int` | Level to print (1-4) |
    | `arg2`   | `string` | Message to print |

</div>

---

### `pd.error`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__
    
    If you want to inform errors in PureData console use `pd.error` method. The message is printed in red.
    
    ``` py
    import pd

    def main(arg1):
        try:
            # some wrong arg here ????
        
        except:
            pd.error("This is a not " +
                     "valid operation!")

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Message of the error. |

</div>

## GUI

### `pd.pd_has_gui`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    When using some functions as `pd.show_image` can be useful to know if a GUI is running. This function returns `False` when running PureData from the terminal using `-nogui`.

    ``` py
    import pd
	    

    def thereisGUI():
        return pd.pd_has_gui()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There are no Arguments.

</div>


### `pd.get_patch_zoom`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    To set different default images for `pd.VIS` objects, you can use this function. It returns 1 when the patch is used without zoom and 2 when is used with zoom. 

    ``` py
    import pd
	    

    def getZoom():
        return pd.get_patch_zoom()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There are no Arguments.

</div>

## Objects

These functions are used to retrieve information about the current object where the function is being executed.

### `get_outlet_count`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Get the number of outlets of the object.
    
    ``` python
    import pd
	    
    def get_out_count():
        pd.get_outlet_count()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no args for this function.

</div>

### `get_obj_args`

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    Returns a list of all arguments used to create the object.
    
    ``` python
    import pd
	    
    def get_out_count():
        pd.get_obj_args()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no args for this function.

</div>

