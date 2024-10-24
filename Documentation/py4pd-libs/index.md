---
hide:
  - navigation
  - toc
---

!!! danger "Nerd stuff"
    This is a section for people that run Python, check how to use the [`py4pd`](../pd-users/py4pd-module.md) library instead.
    
    
To create `py4pd` libraries, the process is basically create a new Github Repository and add one `pyproject.toml` file on it.

``` toml
[project]
name = "py4pd-upic"
version = "0.1.0"
description = "Description."
dependencies = [
    "svgpathtools",
]

[project.urls]
"Source Code" = "https://github.com/username/package" 

[tool.setuptools]
packages = ["mypackage"]  # Name of the package to be imported

# to use subfolders
#"mypackage" = ["*", "othersfiles/*"]  
```

All the scripts must be inside `mypackage` folder. Any script live outside of it. Basically, the object must have this structure. Download the `hello world` example [here](hello_world.zip)!.

```
├─ GITHUB_REPOSITORY
├── mypackage/
    ├── __init__.py
    ├── mysubmodule.py
    └── help/
        ├── object1-help.pd
        └── object2-help.pd
├── pyproject.toml
├── README.md
└─── LICENSE
```


To install this library you must create a `py4pd` object and the a message with: `pip install git+https://github.com/<username>/<repository>`.

<p align="center">
    <img align="center" src="install.png" width="50%" style="border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);"></img>
</p>
