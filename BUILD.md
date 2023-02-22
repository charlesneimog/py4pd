## Building from Source

⚠️ This readme use version `python 3.11` as example, replace all pathnames using the python version you want. It is possible to build it using Github Actions. For that, fork the py4pd repo, go to actions and run it ⚠️.

#### Windows OS 
* replace `<username>` for your username.

``` bash 
pacman -S make autoconf automake libtool mingw-w64-x86_64-libwinpthread-git mingw64/mingw-w64-x86_64-gcc
make PYTHON_INCLUDE="C:/Users/<username>/AppData/Local/Programs/Python/Python311/include" PYTHON_DLL="C:/Users/<username>/AppData/Local/Programs/Python/Python311/python311.dll"
```
OBS.: **Important**, Do not use the `Python311.dll` of `miniconda`, `conda` or other to compile `py4pd`. With the 'original' dll it is possible to replace where the `python311.dll` look for dynamics libraries.

-----------------
If you want to build it on Linux:

#### Linux - version 3.11
* First you need to run: 
    `sudo add-apt-repository ppa:deadsnakes/ppa`.
    `sudo apt-get install python3.11-dev`.

``` bash 
make PYTHON_INCLUDE=/usr/include/python3.10/ PYTHON_VERSION=python3.10 
```

#### MacOS - version 3.11
* First you need to install Python 3.11 (https://www.python.org/downloads/release/python-3110/) then run:

``` bash 
sudo ln -s /Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib /usr/local/lib/libpython3.11.dylib
make PYTHON_INCLUDE=/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11 PYTHON_VERSION=python3.11
```
