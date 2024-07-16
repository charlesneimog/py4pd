# <h2 align="center"> **Py4pd Config File** </h2>

The Py4pd configuration file (py4pd.cfg) can be created in the directory where the py4pd binary is installed. Below is an example configuration file:

```bash
conda_env_packages = /home/neimog/.config/miniconda3.dir/envs/composition/lib/python3.11/site-packages/
editor_command = wezterm -e nvim %s +%d
```

This configuration file enables the customization of additional settings such as Conda environment packages, editor commands, and the choice of editor.

# <h4 align="center"> **Conda Env Packages** </h4>

This feature enables the utilization of packages installed within the Conda environment. As a result, there is no need to install certain packages twice.

# <h4 align="center"> **Editor Commands** </h4>

As shown in [Editor Options](./args.md#editor-options), `py4pd` has five pre-configured editors. `vscode`, `nvim`, `emacs`, `sublime`, and `gvim`. But, for example, with `nvim` on Linux Gnome, `py4pd` assumes that `gnome-console` or `gnome-terminal` is installed.As showed in [Editor Options](./args.md#editor-options), `py4pd` has five pre-configured editors. `vscode`, `nvim`, `emacs`, `sublime` and `gvim`. But, for example, with `nvim` on Linux Gnome, `py4pd` assumes that `gnome-console` or `gnome-terminal` is installed. 

I don't use either of these, I use `wezterm`, so I need to set my editor command as I did. You should research how to open your editor. You need two arguments expressed by `%s` and `%d`. `%s` will be replaced by the file name of the Python Scripted loaded. `%d` will be replaced by line number where the function line is defined. If you can't open your editor in a specified line, you can omit the `%d` argument. 

!!! danger
    The command `%s` always should be used before `%d`. If you change the order this will probably crash PureData.

# <h4 align="center"> **Editor** </h4>

Here you set you default editor. It can be `vscode`, `nvim`, `emacs`, `sublime` or `gvim`.

