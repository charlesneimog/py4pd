import pd
import os
import sys
import platform
import subprocess

def pipinstall(package):
    """Install a Python package from Pd"""
    version = sys.version_info
    major = version.major
    minor = version.minor
    try:
        if isinstance(package, list):
            pd.print('Installing ' + package[1] + ' , please wait...')
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
        home = pd.home()
        # If linux or macos
        if platform.system() == 'Linux':
            from tkinter import Tk, LabelFrame, Label
            root = Tk()
            root.after(1, lambda: root.focus_force())
            try:
                root.title("Installing " + package)
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
                if value != 0:
                    pd.error("Some error occur with Pip.")
                    root.destroy()
                    return 'bang'
                pd.print("Installed " + package)
                pd.error("You need to restart PureData")
                root.destroy()
                return 'bang'
            except Exception as e:
                pd.error(str(e))
                root.destroy()
                return 'bang'
        
        elif os.name == 'nt': 
            folder = f'{folder}/py-modules'
            command = ['py', f'-{major}.{minor}', '-m', 'pip', 'install', '--target', f"{folder}", package, '--upgrade']
            result = subprocess.run(command, check=True)
            if result != 0:
                pd.error("Some error occur with Pip.")
                return 'bang'

        elif platform.system() == 'Darwin':
            try:
                pipmain(['install', '--target', f'{folder}/py-modules', package, '--upgrade'])
                pd.print("Installed " + package)
                pd.print("I recommend restart PureData...")
                return 'bang'
            
            except Exception as e:
                pd.error(str(e))
                return 'bang'

        else:
            pd.error("Your OS is not supported")
            return 'bang'
    except Exception as e:
        pd.print(str(e))
        return 'bang'
