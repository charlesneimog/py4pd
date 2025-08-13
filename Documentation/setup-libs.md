---
hide:
  - navigation
  - toc
---
After creating objects with `py4pd`, you can add them to a library and install it using `pip` or the `py.pip` interface provided by `py4pd`. The process is straightforward: create a new GitHub repository and include a `pyproject.toml` file. This will make the library accessible within `py4pd`. Below is a simple example:

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

TODO: Needs updates

