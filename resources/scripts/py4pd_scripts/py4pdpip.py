
import pd
import os
import sys
import platform
import subprocess

package = ""
folder = ""


class MacOSpip:
    def __init__(self, pippackage, pipfolder):
        self.package = pippackage
        self.folder = pipfolder
        self.window = tk.Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.pipinstall = self._pipinstall  # Renamed to avoid naming conflict
        
    def _pipinstall(self):
        major = version.major  # Assuming version is defined
        minor = version.minor  # Assuming version is defined
        folder = self.folder
        package = self.package
        pd.print(folder)  # Assuming pd is imported and defined
        pd.print(package)  # Assuming pd is imported and defined
        value = subprocess.run([f'/usr/local/bin/python{major}.{minor}', '-m', 'pip', 'install', '--target', f"{folder}/py-modules", package, '--upgrade'], check=True)
        return True
    
    def close_window(self):
        self.window.destroy()
    
    def run(self):
        self.window.mainloop()


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
                folder = pd.home()
            elif folder == "global":
                folder = pd.py4pdfolder() + "/resources"
            else:
                pd.error("[py.install]: the first argument must be 'local' or 'global'")
                return 'bang'
            package = package[1]
        else:
            pd.error("[py.install]: bad arguments")
            return 'bang'
        # If linux or macos
        if platform.system() == 'Linux':
            from tkinter import Tk, LabelFrame, Label
            root = Tk()
            root.after(1, lambda: root.focus_force())
            try:
                # get screen width and height
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()

                # calculate x and y coordinates for the Tk root window to center it on the screen
                x = (screen_width/2) - (300/2)
                y = (screen_height/2) - (100/2)

                root.geometry("300x100+%d+%d" % (x, y))
                root.resizable(False, False)

                # create text
                text = LabelFrame(root, text="Installing " + package + " , please wait...",
                                  padx=20, pady=20)
                text.pack(fill="both", expand=1)

                # add label inside the label frame
                label = Label(text, text="Installing " + package + " , please wait...",
                              anchor="center", justify="center")
                label.pack(fill="both", expand=1)

                # update window
                root.update()   
                value = subprocess.run([f'python{major}.{minor}', '-m', 'pip', 'install', '--target', f"{folder}/py-modules", package, '--upgrade'], check=True)
                if value.returncode != 0:
                    pd.error("Some error occur with Pip.")
                    root.destroy()
                    return 'bang'
                pd.print("Installed " + package)
                pd.error("You need to restart PureData!")
                root.destroy()
                return 'bang'
            except Exception as e:
                pd.error(str(e))
                root.destroy()
                return 'bang'
        
        elif os.name == 'nt': 
            # try to update pip
            command = ['py', f'-{major}.{minor}', '-m', 'pip', 'install', '--upgrade', 'pip']
            result = subprocess.run(command, check=False)
            folder = f'{folder}/py-modules'
            command = ['py', f'-{major}.{minor}', '-m', 'pip', 'install', '--target', f"{folder}", package, '--upgrade']
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                pd.error("Some error occur with Pip.")
                return 'bang'
            pd.print("Installed " + package)
            pd.error("You need to restart PureData!")
            return 'bang'

        elif platform.system() == 'Darwin':
            #import turtle
            import tkinter as tk
            try:
                my_window = MacOSpip(package, folder)
                my_window.run()
                

                
            except Exception as e:
                pd.error(str(e))



        else:
            pd.error("Your OS is not supported")
            return 'bang'
    except Exception as e:
        pd.print(str(e))
        return 'bang'
