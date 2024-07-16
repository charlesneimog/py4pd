## `pd.get_sample_rate`

This gets the current sample-rate of PureData. You can use the `pd.SAMPLERATE` variable too.

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

This gets the current vector-size/blocksize of PureData. This gets the vector size of the object, so it is inside some patch with `block`~ 128`, and the PureData is configured with `vectorsize = 64` it will return 128. To get the PureData vector size you can use `pd.VECSIZE`.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns a unique string.
    
    ``` py
    import pd
	    
    def getVecSize():
        return pd.get_vec_size()


    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>

## `pd.get_num_channels`

This function gets the actual number of channels of the object.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns a unique string.
    
    ``` py
    import pd
	    
    def getNumberOfChannels(audio):
        return pd.get_num_channels()

    ```

-   :fontawesome-solid-gear: __Arguments__

    There is no `args` for this function.
    
</div>
