## `pd.get_sample_rate`

This get the current samplerate of PureData. You can use the `pd.SAMPLERATE` variable too.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string.
    
    ``` python
    import pd
	    
    def getsampleRate():
        return pd.get_sample_rate()

    ```


-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.

    
</div>

---

## `pd.get_vec_size`

This get the current vectorsize/blocksize of PureData. This get the vector size of the object, so it is inside some patch with `block~ 128` and the PureData is configured with `vectorsize = 64` it will return 128. To get the PureData vector size you can use `pd.VECSIZE`.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string.
    
    ``` py
    import pd
	    
    def getVecSize():
        return pd.get_vec_size()


    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>

