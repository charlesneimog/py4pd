These are the methods used to send data from Python to PureData. The inverse path is done mainly with `run` and `key` messages. 
<br>
--------------------------------------
## `pd.out`

`#!python pd.out()` allows you to output data to PureData without needing to wait for the Python function to finish executing. This is different from returning data to PureData using the `#!python return` statement, which requires the function to complete before sending data. 


<div class="grid cards" markdown>


-   :fontawesome-brands-python: __Example__

    For example, consider the following function:

    ``` py
    import pd


    def example_pdout():
        for x in range(10):
        	pd.out(x, symbol="loop") 
            # output in outlet 0

        pd.out("fim", symbol="end", out_n=1) 

    ```
        
-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `Python Object` | Python thing that will be outputed. |

    --- 

    | Kwargs | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `symbol`   | `string` | It prepend the string in the output, can be used with `route` object. |
    | `out_n`   | `int` | Index of the outlet used to output. |

</div>

---------------------------

## `pd.send`

You can use `pd.send` to send data to a receive object in your PureData patch. This method takes in two arguments: the name of the `receive` object and the value you want to send. For instance, suppose you have a receive object named "myReceiver" in your patch. To send the value 42 to this object, you could use `pd.send("myReceiver", 42)`.


<div class="grid cards" markdown>


-   :fontawesome-brands-python: __Example__

    ``` python
    import pd


    def pd.send():
        """It sends a message to 
        the py4pdreceiver receive."""	

	    pd.send("py4pdreceiver", "hello from python!")
	    pd.send("py4pdreceiver", 1) 
	    pd.send("py4pdreceiver", [1, 2, 3, 4, 5])
	    return 0

    ```

    ??? info
        In this example, it will send to `py4pdreceiver` the message `"hello from python!"`, then the number `1`, then the list `[1, 2, 3, 4, 5]`. 

        
-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Name of the receiver object. |
    | `arg2`   | `Python Object` | Data that will be sent. |

</div>


-------------------------------------- 
## `pd.tabwrite`

`pd.tabwrite` is a method that is essentially a copy of the `tabwrite` object in PureData. With this method, you can write audio or any data supported to PureData array.


<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__

    ``` python
    import pd
    import numpy as np

    def randomNumber(len):
        randomNumbers = np.random.rand(len)
        pd.tabwrite("table2test", 
                    randomNumbers, resize=True)
      
    ```

-   :fontawesome-solid-gear: __Arguments__

    
    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Name of the table. |
    | `arg2`   | `Python Object` | List or array (numpy) of numbers. |

    ---
    | Kwargs | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `rezise`   | `Boolean` | Set if the table will be resized or not. |

</div>




-------------------------------------- 
## `pd.tabread`

`pd.tabread` is a method that is essentially a copy of the `tabread` object in PureData. With this method, you can read data from a PureData array directly from within your Python code. It will return one Numpy Array with the data of the table.

<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__


    ``` python
    import pd
    import numpy

    def readFromArray():
        valuesFromArray = pd.tabread("py4pdArray", 
                                     numpy=True)
        multi = numpy.multiply(valuesFromArray, 2)
        return multi

    ```

-   :fontawesome-solid-gear: __Arguments__

    | Parameters | Type    | Description                  |
    | :--------: | :-----: | :--------------------------: |
    |   `arg1`   | `string`|    Name of the table.        |

    ---
    | Kwargs | Type    | Description                  |
    | :--------: | :-----: | :--------------------------: |
    |   `numpy` | `Boolean`| Return a list instead of a numpy array when `False`. |
    

</div>


