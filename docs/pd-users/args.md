# <h2 align="center"> **Arguments** </h2>

---

For the creation of the object, there are some options. Here I will explain each one.

!!! note warning
These arguments are an essential part of py4pd. Not understanding it can generate problems, instabilities, and crash Pd.

## <h3 align="center"> **Editor options** </h3>

For who works with Python, you can set the IDE `editor` in the creation of the `py4pd`. For now, we have four implemented IDEs:

- `-vscode`: It is the default IDE, you do not need to use `-vscode` at least you have an `py4pd.cfg` file.
- `-nvim`: It sets `nvim` as the editor of `py4pd`.
- `-emacs`: It sets `emacs` as the editor of `py4pd`.
- `-sublime`: It sets `sublime` as the editor of `py4pd`.
- `-gvim`: It sets `gvim` as the editor of `py4pd` ( :octicons-mark-github-16: [fferri](https://github.com/fferri)).

## <h3 align="center"> **Set function** </h3>

You can `load` functions in the creation of the object. For that, you must put the script name and then the function name. The name of the script file always need to be the first. You can use `py4pd -canvas score note`, `py4pd score note -canvas`, but `py4pd note score -canvas` or `py4pd note -canvas score` will not work when the script name is `score.py` and `note` is the function.

---

