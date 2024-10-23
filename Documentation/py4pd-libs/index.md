To create `py4pd` libraries, the process is bassicaly create a new Github Repository and add one `pyproject.toml` file on it.

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

To install this library you must create a `py4pd` object and the a message with: `pip install git+https://github.com/username/package`.

Simple like that!
