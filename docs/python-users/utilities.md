## `pd.get_str_pointer`

When working with audio objects, there are situations where we require global variables or variables that retain their values across different runs. For instance, when creating a Python function to generate a sine wave, we may need a global variable for the phase in order to generate a continuous waveform. However, using Python Global Variables can be problematic when working with multiple objects, as all functions would modify the phase value, potentially overwriting it unintentionally. To address this issue, we introduced the `pd.get_obj_pointer` function, which returns a unique string representing the pointer of the C object. This string is unique for each object and can be utilized in other contexts to locate and retrieve the desired global variable. 

=== "Examples" 
    
    It just returns unique string.
    
    ``` py
    import pd
	    
    print(pd.get_str_pointer())

    ```
    
=== "Parameters" 

    There is no `args` for this function.



--------------------------------------

## `pd.get_obj_var`

When working with audio objects, we have another helpful function called `pd.get_obj_var`. This function serves a similar purpose to `pd.get_obj_pointer`. Here, it creates the variable automatically if it doesn't exist yet.

=== "Examples" 

    In the code snippet below, when we use `#!python pd.get_obj_var("PHASE")`, it retrieves the value of the variable associated with the current running object. If the value hasn't been set yet, it will be initialized to `0.0`.

    ``` python
     
    phase = pd.get_obj_var("PHASE", initial_value=0.0)
            
    ```
    
=== "Parameters" 

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Object Variable Name. |

=== "Keywords" 

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `initial_value`   | `Python Object` | With `initial_value` you can set one initial value for the string. | 



--------------------------------------

## `pd.set_obj_var`

To set new values for the variable of the object we use `pd.set_obj_var`. In audio objects, for example, this value you be saved for the next block (next run) calculation.

=== "Examples"

    ``` python
    pd.set_obj_var("PHASE", phase)
    ```
    
=== "Parameters" 

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Object Variable Name. |
    | `arg2`   | `Python Object` | Any Python Object. |

