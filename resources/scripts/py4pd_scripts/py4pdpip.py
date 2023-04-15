import pd
import os

try:
    from pip._internal.cli.main import main as pipmain
    addpip = True
except Exception as e:
    pd.error("Error importing pip: " + str(e))
    addpip = False


def pipinstall(package):
    """Install a Python package from Pd"""
    if isinstance(package, list):
        pd.print('Installing ' + package[1] + ' , please wait...')
        folder = package[0]
        if folder == "local":
            folder = pd.home()
        elif folder == "global":
            folder = pd.py4pdfolder() + "/resources"
        else:
            pd.error("[py.install]: the first argument must be 'local' or 'global'")
            return None
        package = package[1]
    else:
        pd.error("[py.install]: bad arguments")
        return None
    home = pd.home()
    # If linux or macos
    if os.name == 'posix':
        try:
            pipmain(['install', '--target', f'{folder}/py-modules', package, '--upgrade'])
            pd.print("Installed " + package)
            return None
        except Exception as e:
            pd.error(str(e))
            return None
    
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
            return None
    else:
        pd.error("Your OS is not supported")
        return None
