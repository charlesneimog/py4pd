

# py4pd 
#### [Download](https://github.com/charlesneimog/py4pd/releases)


py4pd allows the use of Python inside PureData. It has the same objective as py/pyext being much simpler to keep compile, mainly for Windows OS. It was, in first place, a trying to compile py/pyext for Windows OS, but I couldn't. Then I did the object. It is based in samples of code from Internet and, of course, the code of py-pyext, mainly in the fork of SOPI research group.
1. [How to call Python from C](https://stackoverflow.com/questions/1056051/how-do-you-call-python-code-from-c-code);
2. [C api from Python Docs](https://docs.python.org/3/extending/embedding.html);


## Documentation

#### Using Python Modules

The py4pd is builded to work with site-packages from `venv` (on Windows venv/Lib/site-packages), but it will work with conda too. For example, if my enviroment is installed in `C:/Users/Neimog/Git/py4pd/py4pd/`, you can sent a message `packages  C:/Users/Neimog/Git/py4pd/py4pd/Lib/site-packages` that you will be able to use all modules installed in the enviroment.

#### Run 

You need to define functions in the `.py` file to use it in PureData. For example, let's say that I want to build a Diamond Tonality from Harry Partch. Then I define the `otonal_diamond` function in the `partch.py` script inside the folder where my Pd Patch is saved. 

``` Python
def otonal_diamond(limit, diagonal, fundamental):
    otonal = []
    for o in range(1, limit, 2):
        tonality = []
        for u in range (1, limit, 2):
            tonality.append(fundamental * (u / o))
        otonal.append(tonality)
        tonality = [] 
    return otonal[diagonal]
```

Then a sent the message `set {script_without_.py} {function_name}`, in this case, `set partch otonal_diamond` and run it using `args 13 3 440`. That means that the object will return a Diamond limite 13, the third Diagonal (in this case not the same that the figures), with the A4 how fundamental.

![image](https://user-images.githubusercontent.com/31707161/179780465-0bec0a51-8bdb-4733-a846-7e1952311277.png)
