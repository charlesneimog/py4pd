import pd
import os
from pip._internal.cli.main import main as pipmain


def install(package):
    pd.print('Installing ' + package + ' , please wait...')
    home = pd.home()
    # If linux or macos
    if os.name == 'posix':
        try:
            pipmain(['install', '--target', f'{home}/py-modules', package, '--upgrade'])
            pd.print('Installed ' + package + ' to ' + home + '/py-modules')
        except Exception as e:
            pd.error(str(e))
    
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




