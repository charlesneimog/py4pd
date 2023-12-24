In `py4pd`, sometimes you need to save something in the patch folder (as audio files), or save something temporally, or anothers things. In this sections we present some helpers to get the correct pathname that can be usefull.


## `pd.get_patch_dir`

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


## `pd.get_temp_dir`



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

## `pd.get_py4pd_dir`

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


## `pd.get_pd_search_paths`



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
