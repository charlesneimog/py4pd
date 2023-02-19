# py4pd 

<div>

py4pd allows the use of Python inside PureData. It has the same objective as py/pyext being much simpler to keep compile, mainly for Windows OS. It was, in first place, a trying to compile py/pyext for Windows OS, but I couldn't. Then I did the object. It is based in samples of code from Internet and how py-pyext works, mainly in the fork of SOPI research group.
1. [How to call Python from C](https://stackoverflow.com/questions/1056051/how-do-you-call-python-code-from-c-code);
2. [C api from Python Docs](https://docs.python.org/3/extending/embedding.html);

<div>

#### You must to install Python.
To download the object: Open PureData, `Help->Find Externals->` search for `py4pd`.

* Linux: `sudo dnf install python3.11` or `sudo apt install python3.11`;
* MacOS: Go to https://www.python.org/downloads/release/python-3112/ and install normally.
* Windows: Go to https://www.python.org/downloads/release/python-3112/ and install normally.

## Wiki

* If you want to use, read the wiki page: https://github.com/charlesneimog/py4pd/wiki


## Building


#### Windows OS - replace `<username>` for your username.

``` bash 
pacman -S make autoconf automake libtool mingw-w64-x86_64-libwinpthread-git mingw64/mingw-w64-x86_64-gcc
make PYTHON_INCLUDE="C:/Users/<username>/AppData/Local/Programs/Python/Python310/include" PYTHON_DLL="C:/Users/<username>/AppData/Local/Programs/Python/Python310/python310.dll"
```
OBS.: **Important**, Do not use the `Python310.dll` of `miniconda`, `conda` or other to compile `py4pd`. With the 'original' dll it is possible to replace where the `python310.dll` look for dynamics libraries.

-----------------
If you want to build it on Linux:

#### Linux - version 3.10
* First you need to run: 
    `sudo add-apt-repository ppa:deadsnakes/ppa`.
    `sudo apt-get install python3.10-dev`.

``` bash 
make PYTHON_INCLUDE=/usr/include/python3.10/ PYTHON_VERSION=python3.10 
```

#### MacOS - version 3.10
* First you need to install Python 3.10 (https://www.python.org/downloads/release/python-3108/) then run:

``` bash 
sudo ln -s /Library/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib /usr/local/lib/libpython3.10.dylib
make PYTHON_INCLUDE=/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10 PYTHON_VERSION=python3.10
```



