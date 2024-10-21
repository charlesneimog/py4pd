In our last project, the object's data type wasn't specified, meaning any input (number, word/symbol, or list) triggered the same function. However, sometimes we need different actions for different inputs. We can achieve this using the `new_object` function.

The `pd.new_object` function returns a class where we can define specific functions for numbers, symbols, or lists. Additionally, we can add custom methods (messages in Pd) to the object. This allows us to adapt the object's behavior to different types of input and create more specialized functionalities.

See for example the library `freesound`, in this library we want to be able to download sounds using the FreeSound api. For that, we need to create one object to search and download the sounds. But before that, we also need to be logged in the freesound account. Check how we create all these thing in one object.

``` py
def py4pdLoadObjects():
    # freesound
    pd_freesound = pd.new_object("freesound")
    pd_freesound.py_out = True
    pd_freesound.ignore_none = True

    # login
    pd_freesound.addmethod("set", set_login_var)
    pd_freesound.addmethod("oauth", initialize_oauth)
    pd_freesound.addmethod("login", login)

    # search
    pd_freesound.addmethod("target", target)
    pd_freesound.addmethod("filter", filter)
    pd_freesound.addmethod("query", query)
    pd_freesound.addmethod("search", search)
    pd_freesound.addmethod("clear", clear)  # clear all configs

    # get info and download
    pd_freesound.addmethod("get", get)
    pd_freesound.addmethod("download", download)

    pd_freesound.add_object()
```

In this object we have a lot of methods, some to do the login steps, anothers to search the audio, to get the info and download it. This is a basic work that can be done using the `pd.new_object` method. Check the help of the [py4pd-freesound](https://github.com/charlesneimog/py4pd-freesound) project to test this.


