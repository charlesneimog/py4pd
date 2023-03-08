# PureData Users


## Valid Messages for `py4pd`

Next, I present all methods used in `pd` module. 

-------------------------------------- 
### `set` 

Set Python Function to `py4pd` object.

* `arg1`: name of the script/library
* `arg2`: name of the function
* `example`: `set score chord`.

_Obs.:_ If you will always use the save function, you can create the object with the `py4pd script function`. 

-------------------------------------- 
### `run` 

Run the Python function. You must use `set` first.

* `*arg`: The list of args will change according to the function.
* `example`: `run 1 2`, `run [c4 c5 db6]` and others.

-------------------------------------- 
### `key` 

Set value for some `string` key. This value is stored inside `py4pd` object and can be used inside Python using `pd.getkey`.

* `arg1`: Name of the `key`.
* `arg2`: Value of the `key`.
* `example`: `key fft-size 1024`, `key clef G`, `key primenumbers [11, 19, 1997]`.

-------------------------------------- 
### `doc` 

It prints on PureData console the documentation of the function.

* `arg`: This message do not use any args.
* `example`: `doc`.

-------------------------------------- 
### `open` 

It open script files, case the file does not exist in the patch folder, it creates a new Python Script. `open score`, for example, will open the `score.py` or create and `score.py`.

* `arg1`: name of the script
* `example`: `open score`.

-------------------------------------- 
### `editor` 

With any arguments it will open the Python Script loaded with the message `set`. Additionally you can choose between four IDE: `vscode`, `nvim`, `emacs` and `sublime`.

* `arg1`: name of the editor that you use.
* `example`: `editor nvim`.

-------------------------------------- 
### `reload` 

If you are working on a Python Script and change the code, you need to reload it.

* `arg`: There is no args. 
* `example`: `reload`.

-------------------------------------- 
### `restart` 

This restart the Python for all objects `py4pd`.

* `arg`: There is no args. 
* `example`: `restart`.

-------------------------------------- 
### `thread` 

This turn on/off the threads of Python.

* `arg1`: `1` for threads `on` `0` for threads `off`. 
* `example`: `thread 0`. 
* `obs`: I remove this functions because of problems. I will wait for the PEP 684 (search on Google).

-------------------------------------- 
### `numpy` 

This turn on/off the numpy arrays in audio functions

* `arg1`: `1` for numpy `on` `0` for numpy `off`. 
* `example`: `numpy 1`. 

_Obs.:_ The use `numpy` make the code more fast, about 30%.

-------------------------------------- 
### `home` 

Set the home for Python. 

* `arg1`: The new `home` pathname.
* `example`: `home ~/Documents/Git/`. 

-------------------------------------- 
### `packages` 

Set the packages path for Python. `py4pd` will look for modules inside this folders.

* `arg1`: The new `packages` pathname.
* `example`: `packages /home/neimog/miniconda3/envs/composition/lib/python3.11/site-packages`. 
