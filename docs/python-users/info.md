There are three messages used to print info in the PureData console, `pd.print`, `pd.logpost` and `pd.error`.

------------------
## `pd.print`

The ordinary function `#!python print()` will not work in `py4pd` (unless that you open PureData from the terminal). So if you want to debug or print some info from the PureData console you need to use `#!python pd.print`. 



<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
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
    | `**show_prefix`   | `Python Object` | When `False` don't print prefix |

    !!! info
        For Object written in Python, prefix will be the Object Name.
    
</div>

------------------
## `pd.logpost`

This function uses `logpost` in C PureData API to log messages using levels. For example, if you use `#!python logpost(4, "PureData message in level 4")`, the message will appear in console just if the user had selected to show the messages of level 4 in PureData Console.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    It just returns unique string with the pathname.
    
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

## `pd.error`

If you want to inform errors in PureData console use `pd.error` method. 


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__
    
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

