# PureData Object Usage

## <h2 align="center"> **`py4pd` Args** </h2>
-------------------------------------- 

For the creation of the object, there is some options. Here I will explain each one.

!!! note warning
	These arguments are an essential part of py4pd. Not understanding it can generate problems, instabilities, and crash Pd. 
    
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


#### <h3 align="center"><code>set</code></h3> 

It set/load Python Functions to `py4pd` object.

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `symbol` | Python Script name (never uses |
    | `arg2`   | `symbol` | String to create the object. |
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/set.png" width="40%" alt="Set method example">
    </p>

    ??? tip "Tip"
	    If you will always use the same function, you can create the object with the `py4pd script function`. 
	    
-------------------------------------- 
#### <h3 align="center"> <code>run</code> </h3> 


=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `symbol` | Python Script name (never uses |
    | `arg2`   | `anything` | Args for the function |
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/run.png" width="40%" alt="Set method example">
    </p>

    !!! info "Info"

	    The function must be loaded. 

-------------------------------------- 
#### <h3 align="center"> <code>key</code> </h3> 

Set value for some `string` key. This value is stored inside `py4pd` object and can be used inside Python using `pd.getkey`.

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `args`   | `anything` | Args for the function |
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/key.png" width="40%" alt="Set method example">
    </p>

----------------------------- 
#### <h3 align="center"> <code>doc</code> </h3> 

It prints on PureData the documentation of the Python Function (if it exits).
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/doc.png" width="40%" alt="Set method example">
    </p>
    
--------------------------------------

### <h3 align="center"> **Developer Methods** </h3>

-------------------------------------- 

#### <h3 align="center"> <code>open</code> </h3> 

It opens `py` script files, case the file does not exist in the patch folder, it creates a new Python Script. `open score`, for example, will open the `score.py` (if it exists) or create `score.py`

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `args`   | `symbol` | script file name **without** extension `.py`. |
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/open.png" width="40%" alt="Set method example">
    </p>

-------------------------------------- 
#### <h3 align="center"> <code>editor</code> </h3> 

Without arguments it opens the Python Script loaded with the message `set`. With one symbol args you can choose between four IDE: `vscode`, `nvim`, `emacs` or `sublime`. The function must be loaded first.

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `symbol` | `vscode`, `nvim`, `emacs` and `sublime`. |
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/editor.png" width="40%" alt="Set method example">
    </p>

    ??? tip "Tip"
	    If one function is loaded, `click` in the object will open the loaded function too.

-------------------------------------- 
#### <h3 align="center"> <code>reload</code> </h3> 

If you are working on a Python Script and changing the code, you need to send this message to the `py4pd` for the changes to be loaded. 
    
=== "Example"
    
    <p align="center">
        <img src="../examples/pd-methods/reload.png" width="40%" alt="Set method example">
    </p>

-------------------------------------- 
#### <h3 align="center"> <code>home</code> </h3> 

Set the home for Python. It is similar to execute Python from some specific folder. For example, when we use `cd Downloads` then `python myscript.py` in the same terminal.

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `symbol` | Folder that will be the `HOME` for Python Script. |
    
=== "Example"

     <p align="center">
        <img src="../examples/pd-methods/home.png" width="40%" alt="Set method example">
    </p>
    
-------------------------------------- 
#### <h3 align="center"> <code>packages</code> </h3> 

Set the packages path for Python. `py4pd` will look for external modules inside this folders. For example, if you one virtual enviroment called `composition` with miniconda, you can send `packages /home/~USER~/miniconda3/envs/composition/lib/python3.11/site-packages` to use the installed packages.

=== "Args"

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `symbol` | Folder that will be the `HOME` for Python packages. |
    
=== "Example"

     <p align="center">
        <img src="../examples/pd-methods/packages.png" width="60%" alt="Set method example">
    </p>

