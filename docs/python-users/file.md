## `pd.get_patch_dir`

In `py4pd`, `pd.get_home_folder` is a function that returns the path to the directory where the currently-running PureData patch is located. This can be useful for accessing files and resources relative to the location of the patch. 

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
    ``` py
    import pd
	    
    def getPatchDir():
        return pd.get_patch_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>


-------------------------------------- 


## `pd.get_temp_dir`

`pd.get_temp_dir` returns one pathname to save stuff that won't be used more than once, all files inside this folder are deleted when the PureData patch is closed.


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
    ``` py
    import pd
	    
    def getTempDir():
        return pd.get_temp_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>

---

## `pd.get_py4pd_dir`

`pd.get_py4pd_dir` returns the folder where the binary of `py4pd` is located.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
    ``` py
    import pd
	    
    def getPy4pdDir():
        return pd.get_py4pd_dir()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>


