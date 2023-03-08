# Embedded Module

For those using Python, it is possible to communicate between Python and PureData using some of the functions in the Pd module. The Pd module is embedded in the py4pd code and accessible only in the PureData environment. It is similar to what is used inside Google Collab like `google.colab.drive`, `google.colab.widgets`, and others.

For example, to write to a PureData array you can use the method called `pd.tabwrite`, which accepts the array name and one list or numpy array and a keyword (`resize=`) where you resize or not the table. 

``` py
import pd
from random import randint

def pd_tabwrite():
    "It writes data on the pd.tabwrite array."
    randomNumbers = []
    tablen = randint(10, 200)
    i = 0
    while i < tablen:
        # gerar aleatoric number between -1 and 1
        randomNumbers.append(randint(-100, 100) / 100)
        i += 1
    pd.tabwrite("pd.tabwrite", randomNumbers, resize=True) # (1)!

```

1.  There should be an array called `pd.tabwrite` in Patch.


This will write the list `randomNumbers` in the `pd.tabwrite` table in PureData. If the table not exist it will give an error, like happens in tabwrite object.





## Table of Embedded Method

--------------------------- 
* [pd.out](###pd.out) - Output in PureData from any place in Python Code 
* [pd.send](###pd.send) - Send data to PureData, it is received with `receive` object.
* [pd.print](###pd.print) - Print in PureData console.
* [pd.tabwrite](###pd.tabwrite) - Write data in PureData arrays.
* [pd.tabread](###pd.tabread) - Read PureData arrays.
* [pd.show](###pd.show) show images in PureData canvas.
* [pd.home](###pd.home) - Get the current directory of the PureData Patch.
* [pd.tempfolder](###pd.tempfolder) - Get the tempfolder directory of `py4pd`. It's always clean.
* [pd.getkey](###pd.getkey) - Get keys saved with `key` message in `py4pd` object.
* [pd.samplerate](###pd.samplerate) - Get the current Sample Rate of PureData
* [pd.vecsize](###pd.vecsize) - Get current vector size of PureData.
-------------------------

## Methods description

### `pd.out` 

With this object you can output things without the function finish your work. For example, given this function:

``` Python
import pd


def example_pdout():
    for x in range(10):
    	pd.out(x)
    return x
```
it will output 1, 2, 3, (...) like in `else/iterate`. 

---------------------------

### `pd.send` 

With `pd.send` you can send data for `receive` object in PureData Patch. It accepts two arguments, the `receive` name and the value that will be sent. For example, 
``` python
import pd


def pd_send():
    "It sends a message to the py4pdreceiver receive."	
	pd.send("py4pdreceiver", "hello from python!")
	pd.send("py4pdreceiver", 1) # (1)! 
	pd.send("py4pdreceiver", [1, 2, 3, 4, 5])
	return 0

```


### `pd.print` 

### `pd.error` 

### `pd.tabwrite` 

### `pd.tabread`
 
### `pd.show`

### `pd.home`

### `pd.tempfolder`

### `pd.getkey`
    
### `pd.samplerate`

### `pd.vecsize`



