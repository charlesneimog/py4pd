import pd
import os
import sys
import platform
import time

try:
    from pip._internal.cli.main import main as pipmain
    pd.print('Modulo achado')
    addpip = True
except Exception as e:
    pd.error(str(e))
    addpip = False
    if os.name == 'nt':
        pd.error("You need to install pip for windows")
    elif os.name == 'posix':
        pd.error("Open one terminal window and run 'sudo apt install python3-pip', or 'sudo pacman -S python-pip', or 'sudo dnf install python3-pip'")
        sys.exit(1)
    elif os.name == 'darwin':
        pd.error("Open one terminal window and run 'sudo easy_install pip'")
        sys.exit(1)

def closeWindow():
    root.destroy()


def pipinstall(package):
    """Install a Python package from Pd"""
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
                pipmain(['install', '--target', f'{folder}/py-modules', package, '--upgrade'])
                pd.print("Installed " + package)
                pd.print("I recommend restart PureData...")
                root.destroy()


                return 'bang'
            except Exception as e:
                pd.error(str(e))
                root.destroy()
                return 'bang'
        
        elif platform.system() == 'Darwin':
            from tkinter import Tk, LabelFrame, Label
            root = Tk()
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
                pipmain(['install', '--target', f'{folder}/py-modules', package, '--upgrade'])
                pd.print("Installed " + package)
                pd.print("I recommend restart PureData...")
                root.destroy()
                return 'bang'
            except Exception as e:
                pd.error(str(e))
                root.destroy()
                return 'bang'

        # If windows
        elif os.name == 'nt':
            import subprocess
            try:
                # change / to \\
                home = home.replace('/', '\\')  
                # install package from pip without opening a new window
                subprocess.run(f'pip install --target {home}\\py-modules {package} --upgrade', shell=True, check=True)
                pd.print("Installed " + package)
                pd.print("I recommend restart PureData...")       
            except Exception as e:
                pd.error(str(e))
                return 'bang'
        else:
            pd.error("Your OS is not supported")
            return 'bang'
    except Exception as e:
        pd.print(str(e))
        return 'bang'
