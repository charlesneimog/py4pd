# PureData Users

## <h2 align="center"> **`py4pd` Args** </h2>
-------------------------------------- 

For the creation of the object, there is some options. Here I will explain each one.

!!! note warning
	This is a very important part of `py4pd`, as users, you need to understand the ideia of all arguments. **Not understandt it can generate problems and instabilities.**
    
-------------------------------------- 
### <h3 align="center"> **Load Python Libraries** </h3>

It is possible to use `py4pd` to create PureData objects using Python. To load this Python objects in PureData you can use `py4pd` with the `-library` flag and then the name of the script. For example, if the script have the name `myPythonObjects.py`, you need to create an `py4pd` object with `py4pd -library myPythonObjects`. 

!!! info "Info"
	The order of the arguments is important, `py4pd myPythonObjects -library` will not work!
        
-------------------------------------- 
### <h3 align="center"> **Audio options** </h3>


If the Python function that you are using work with audio, you need to create the object with one of this argument `-audioin`, `-audioout` or `-audio`.

* `-audioin`: It creates one object that accepts audio input but no audio output, this can be used for audio analysis. Like `sigmund~`.
* `-audioout`: It creates one object that accepts audio output but no audio input, this can be used for create some synthesis process inside Python.
* `-audio`: It creates one object that accepts audio input and output, this is used for create audio manipulation in Python.

-------------------------------------- 
### <h3 align="center"> **Canvas options** </h3>

There are options to show things inside a PureData patch. This can be used for show images, score, graphs and others.

* `-canvas`: It creates one clear canvas where, from Python, you can show things inside PureData. See some examples:
* `-score`: It creates a clear score used to show scores inside PureData. 

??? Tip "Tip"

	This is part of `py4pd` scripts. You can use, `py4pd -score score note` to show one single note, `py4pd -score score chord` to show chords, and 	soon will be possible to use `py4pd -score score chord-seq` to show chord-seq (yes, like OpenMusic and OM-Sharp). 

### <h3 align="center"> **Editor options** </h3>

For who works with Python, you can set the IDE `editor` in the creation of the `py4pd`. For now, we have four implemented IDEs:

* `-vscode`: It is the default IDE, you do not need to use `-vscode` at least you have an `py4pd.cfg` file.
* `-nvim`: It sets `nvim` as the editor of `py4pd`.
* `-emacs`: It sets `emacs` as the editor of `py4pd`.
* `-sublime`: It sets `sublime` as the editor of `py4pd`.

### <h3 align="center"> **Set function** </h3>

You can `load` functions in the creation of the object. For that, you must put the script name and then the function name. The name of the script file always need be the first. You can use `py4pd -canvas score note`, `py4pd score note -canvas` but `py4pd note score -canvas` or `py4pd note -canvas score` will not work when the script name is `score.py` and `note` is the function.

-------------------------------------- 
## <h2 align="center"> **`py4pd` Methods** </h2>

Next, I present all methods used in `py4pd` object. But the three main `methods` are: (1)

* **set**: It load the `Python` function.
* **install**: It install `Python` libraries **(not ready yet)**.
* **run**: It run the `Python` function.
* **key**: It save `parameters` for be used inside functions.


-------------------------------------- 

### <h3 align="center"> **User Methods** </h3>


#### `set` 

It set/load Python Function to `py4pd` object.

* `arg1`: name of the script/library
* `arg2`: name of the function
* `example`: `set score chord`.

??? tip "Tip"
	If you will always use the same function, you can create the object with the `py4pd script function`. 

-------------------------------------- 
#### `run` 

Run the Python function.

* `*arg`: The list of args will change according to the function.
* `example`: `run 1 2`, `run [c4 c5 db6]` and others.

??? info "Info"

	The function must be loaded. 

-------------------------------------- 
#### `key` 

Set value for some `string` key. This value is stored inside `py4pd` object and can be used inside Python using `pd.getkey`.

* `arg1`: Name of the `key`.
* `arg2`: Value of the `key`.
* `example`: `key fft-size 1024`, `key clef G`, `key primenumbers [11 19 1997]`.

-------------------------------------- 
#### `doc` 

It prints on PureData the documentation of the function (if it exits).

* `arg`: This message do not use any args.
* `example`: `doc`.

--------------------------------------

### <h3 align="center"> **Developer Methods** </h3>

-------------------------------------- 

#### `open` 

It open `py` script files, case the file does not exist in the patch folder, it creates a new Python Script. `open score`, for example, will open the `score.py` (if it exists) or create `score.py`.

* `arg1`: name of the script
* `example`: `open score`.

-------------------------------------- 
#### `editor` 

With no arguments it will open the Python Script loaded with the message `set`. Additionally you can choose between four IDE: `vscode`, `nvim`, `emacs` or `sublime`.

* `arg1`: name of the editor that you use.
* `example`: `editor nvim`.

??? tip "Tip"
	If one function is loaded, `click` in the object will open the loaded function too.

-------------------------------------- 
#### `reload` 

If you are working on a Python Script and change the code, you need send this message to the `py4pd` for the changes be loaded. 

* `arg`: There is no args. 
* `example`: `reload`.

-------------------------------------- 
#### `restart` 

This restart the Python for all objects `py4pd`. 

* `arg`: There is no args. 
* `example`: `restart`.

??? warning "Warning"
	Caution, this can crash PureData and gerenerate problem with `import` modules. 

-------------------------------------- 
#### `thread` 

This turn `on/off` the threads of Python.

* `arg1`: `1` for threads `on` `0` for threads `off`. 
* `example`: `thread 0`. 

??? failure "Failure"
	I removed this functions because of problems with the GIL of Python. I will wait for the [PEP 684](https://peps.python.org/pep-0684/) that probably will be part of Python `3.12`.

-------------------------------------- 
#### `numpy` 

This turn `on/off` the numpy arrays in the `input` of audio functions. 

* `arg1`: `1` for numpy `on` `0` for numpy `off`. 
* `example`: `numpy 1`. 

??? Tip "Tip"
	The use `numpy` make the code more fast, about 30% with using `fft` and `ifft` of `numpy`. 

-------------------------------------- 
#### `home` 

Set the home for Python. It is similar to execute Python from some specific folder. For example, use `cd Downloads` then `python myscript.py` in the same terminal.

* `arg1`: The new `home` pathname.
* `example`: `home ~/Documents/Git/`. 

-------------------------------------- 
#### `packages` 

Set the packages path for Python. `py4pd` will look for modules inside this folders.

* `arg1`: The new `packages` pathname.
* `example`: `packages /home/neimog/miniconda3/envs/composition/lib/python3.11/site-packages`. 

??? tip "Tip"
	Observe that you can use `conda` or `miniconda` environment. In this example, I am using my conda environment called `composition`.


