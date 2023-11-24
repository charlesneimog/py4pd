import faulthandler
import os
import platform
import subprocess
import sys

import pd

faulthandler.enable()

package = ""
folder = ""


class MacOSpip:
    def __init__(self, pippackage, pipfolder):
        import tkinter as tk

        self.package = pippackage
        self.folder = pipfolder
        self.window = tk.Tk()
        # icon_file = pd.py4pdfolder() + "/resources/icons/pd.icns"
        # self.window.iconbitmap(icon_file)
        # self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self._pipinstall()  # Renamed to avoid naming conflict

    def drawWarning(self):
        from tkinter import Label, LabelFrame, Tk

        # get screen width and height
        root = self.window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # calculate x and y coordinates for the Tk root window to center it on the screen
        x = (screen_width / 2) - (300 / 2)
        y = (screen_height / 2) - (100 / 2)

        root.geometry("300x100+%d+%d" % (x, y))
        root.resizable(False, False)

        # create text
        text = LabelFrame(
            root, text="Installing " + package + " , please wait...", padx=20, pady=20
        )
        text.pack(fill="both", expand=1)

        # add label inside the label frame
        label = Label(
            text,
            text="Installing " + package + " , please wait...",
            anchor="center",
            justify="center",
        )
        label.pack(fill="both", expand=1)

        # update window
        self.window.update()

    def _pipinstall(self):
        self.drawWarning()
        version = sys.version_info
        major = version.major  # Assuming version is defined
        minor = version.minor  # Assuming version is defined
        folder = self.folder
        package = self.package
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
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.window.after(200, self.close_window)
        self.window.mainloop()

        return True

    def close_window(self):
        self.window.quit()
        self.window.destroy()

    def run(self):
        self.window.update()


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
                my_window = MacOSpip(package, folder)
                my_window.run()

            except Exception as e:
                pd.error(str(e))
        else:
            pd.error("Your OS is not supported")
            return "bang"
    except Exception as e:
        pd.print(str(e))
        return "bang"
