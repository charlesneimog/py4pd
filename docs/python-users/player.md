## `pd.add_to_player`

With this function you create a `Python Object Player`, so you can trigger Python functions using times parameters. See [py4pd-upic](https://github.com/charlesneimog/py4pd-upic), in this library I use `pd.add_to_player` to play svg draws that are used to control parameters.


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string.
    
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

## `pd.clear_player`

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




