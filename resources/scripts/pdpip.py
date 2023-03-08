import pd
from pip._internal.cli.main import main as pipmain

def install(package):
    pd.print('Installing ' + package + ' , please wait...')
    home = pd.home()
    try:    
        pipmain(['install', '--target', f'{home}/py-modules', package, '--upgrade'])    
        pd.print('Installed ' + package + ' to ' + home + '/py-modules')
    except Exception as e:
        pd.error(str(e))




