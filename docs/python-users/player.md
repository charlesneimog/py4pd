For musicians, it is important to put things in time. With the functions presented here you can call specific functions using `onsets` values. For example, in [py4pd-upic](https://github.com/charlesneimog/py4pd-upic) I use `pd.add_to_player` to play svg draws that are used to control parameters. So each `svg` elements trigger some specific function in a specific point in time.

!!! danger "Set the right configuration"
    <p style="font-size: 16px;">To use this `methods` the Python Object must be configurable with `playable` as `#!python True`. See [configuration](pd.new_object/config.md#playable) for `player` object.</p>

## `pd.add_to_player`

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




