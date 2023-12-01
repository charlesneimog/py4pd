import os
import platform
import subprocess
import sys

import pd


package = ""
folder = ""


def pipinstall(mypackage):
    """Install a Python package from Pd"""
    global folder
    global package
    package = mypackage
    version = sys.version_info
    major = version.major
    minor = version.minor
    try:
        if isinstance(package, list):
            folder = package[0]
            if folder == "local":
                folder = pd.get_patch_dir()
            elif folder == "global":
                folder = pd.get_py4pd_dir() + "/resources"
            else:
                pd.error("[py.install]: the first argument must be 'local' or 'global'")
                return "bang"
            package = package[1]
        else:
            pd.error("[py.install]: bad arguments")
            return "bang"
        # If linux or macos
        if platform.system() == "Linux":
            root = None
            try:
                # if pd.pd_has_gui():
                # from tkinter import Label, LabelFrame, Tk
                #
                # root = Tk()
                # root.after(1, lambda: root.focus_force())
                # screen_width = root.winfo_screenwidth()
                # screen_height = root.winfo_screenheight()
                # x = (screen_width / 2) - (300 / 2)
                # y = (screen_height / 2) - (100 / 2)
                # root.geometry("300x100+%d+%d" % (x, y))
                # root.resizable(False, False)
                # text = LabelFrame(
                #     root,
                #     text="Installing " + package + " , please wait...",
                #     padx=20,
                #     pady=20,
                # )
                # text.pack(fill="both", expand=1)
                # label = Label(
                #     text,
                #     text="Installing " + package + " , please wait...",
                #     anchor="center",
                #     justify="center",
                # )
                # label.pack(fill="both", expand=1)
                # root.update()
                value = subprocess.run(
                    [
                        f"python{major}.{minor}",
                        "-m",
                        "pip",
                        "install",
                        "--target",
                        f"{folder}/py-modules",
                        package,
                        "--upgrade",
                    ],
                    check=True,
                )
                if value.returncode != 0:
                    pd.error("Some error occur with Pip.")
                    if root:
                        root.destroy()
                    return "bang"
                pd.print("Installed " + package)
                pd.error("You need to restart PureData!")
                if root:
                    root.destroy()
                return "bang"
            except Exception as e:
                pd.error(str(e))
                if root:
                    root.destroy()
                return "bang"

        elif os.name == "nt":
            # try to update pip
            command = [
                "py",
                f"-{major}.{minor}",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
            result = subprocess.run(command, check=False)
            folder = f"{folder}/py-modules/"
            # change all / to \
            folder = folder.replace("/", "\\")

            command = [
                "py",
                f"-{major}.{minor}",
                "-m",
                "pip",
                "install",
                "--target",
                f"{folder}",
                package,
                "--upgrade",
            ]
            pd.print(command)
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                pd.error("Some error occur with Pip.")
                return "bang"
            pd.print("Installed " + package)
            pd.error("You need to restart PureData!")
            return "bang"

        elif platform.system() == "Darwin":
            try:
                value = subprocess.run(
                    [
                        f"/usr/local/bin/python{major}.{minor}",
                        "-m",
                        "pip",
                        "install",
                        "--target",
                        f"{folder}/py-modules",
                        package,
                        "--upgrade",
                    ],
                    check=True,
                )
                if value.returncode != 0:
                    pd.logpost(3, "pip return value" + str(value))
                    pd.error(
                        "You need to restart PureData, to check if the installation process worked"
                    )
                else:
                    pd.print(f"{package} Installed!")
                    pd.error("You need to restart PureData")
            except Exception as e:
                pd.error(str(e))
        else:
            pd.error("Your OS is not supported")
            return "bang"
    except Exception as e:
        pd.print(str(e))
        return "bang"
