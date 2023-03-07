# Pd Module

For those using Python, it is possible to communicate between Python and PureData using some of the functions in the Pd module. The Pd module is embedded in the py4pd code and accessible only in the PureData environment. It is similar to what is used inside Google Collab like `google.colab.drive`, `google.colab.widgets`, and others.

For example, to write to a PureData array you can use the method called `pd.tabwrite`, which accepts the array name and one list or numpy array and a keyword where you resize or not the table. 

``` Python
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
    pd.tabwrite("pd.tabwrite", randomNumbers, resize=True)

```

This will write the list `randomNumbers` in the `pd.tabwrite` table in PureData. If the table not exist it will give an error, like happens in tabwrite object.

---------------------------

With this examples, the list of avaible function in `pd` module are

## `pd.out` 

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

## `pd.send` 

With `pd.send` you can send data for `receive` object in PureData Patch. It accepts two arguments, the `receive` name and the value that will be sent. For example, 
``` python
import pd


def pd_send():
    "It sends a message to the py4pdreceiver receive."	
	pd.send("py4pdreceiver", "hello from python!")
	pd.send("py4pdreceiver", 1)
	pd.send("py4pdreceiver", [1, 2, 3, 4, 5])
	return 0

```







    {"send", pdsend, METH_VARARGS, "Send message to PureData, it can be received with the object [receive]"},
    {"print", pdprint, METH_VARARGS, "Print informations in PureData Console"},
    {"error", pderror, METH_VARARGS, "Print informations in error format (red) in PureData Console"},
    {"tabwrite", (PyCFunction)pdtabwrite, METH_VARARGS | METH_KEYWORDS, "Write data to PureData tables/arrays"},
    {"tabread", pdtabread, METH_VARARGS, "Read data from PureData tables/arrays"},
    
    // Pic
    {"show", pdshowimage, METH_VARARGS, "Show image in PureData, it must be .gif, .bmp, .ppm"},

    // Files
    {"home", pdhome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"tempfolder", pdtempfolder, METH_VARARGS, "Get PureData Temp Folder"},

    // User
    {"getkey", pdkey, METH_VARARGS, "Get Object User Parameters"},

    // audio
    {"samplerate", pdsamplerate, METH_VARARGS, "Get PureData SampleRate"},
    {"vecsize", pdveczise, METH_VARARGS, "Get PureData Vector Size"},

    {NULL, NULL, 0, NULL}  //







## Add py4pd your library in py4pd Download

## Embedded Module

