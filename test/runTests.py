import os
import subprocess
import sys
import platform

errorInTest = 0


def runTest(pdpatch):
    global errorInTest
    if platform.system() == 'Linux':
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "/" + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            cmd = f'pd -nogui -send "start-test bang" {pathfile}' 
            # print cmd in green
            print("Running: " + "\033[92m" + cmd + "\033[0m")
        else:
            print('PureData Object not found')
            sys.exit()
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        
    elif platform.system() == 'Windows':
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            pass
        else:
            print(f'Patch {pathfile} not found')
            sys.exit()
        os.system(f'cmd /c "\"C://Program Files//Pd//bin//pd.exe\" -send \"start-test bang\" \"{pathfile}\""')
        output = subprocess.run(f'"C:\\Program Files\\Pd\\bin\\pd.exe" -nogui -send "start-test bang" "{pathfile}"', capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
    elif platform.system() == 'Darwin':
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "/" + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            pass
        else:
            print(f'Patch {pathfile} not found')
            sys.exit()

        cmd = '/Applications/Pd-*.app/Contents/Resources/bin/pd -stderr -send "start-test bang" ' + pathfile
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
    else:
        print('OS not supported')
        sys.exit()

    # if lastLine contains "PASS" then the test passed
    passed = False
    for line in outputLines:
        if "PASS" in line or "Pass" in  line:
            passed = True
    if passed:
        print("\033[92m" + ' Test with ' + pdpatch + ' passed' + "\033[0m")
    else:
        print("\033[91m" + ' Test with ' + pdpatch + ' failed' + "\033[0m")
        errorInTest += 1

    
if __name__ == "__main__":
    # list all patches inside test folder
    patches = os.listdir('.')
    patches = [patch for patch in patches if patch.endswith('.pd')]
    patches.sort()
    for patch in patches:
        runTest(patch)

    if errorInTest != 0:
        print("\033[91m" + f'{errorInTest} Test has failed' + "\033[0m")
        sys.exit(-1)

    elif errorInTest == 0:
        print("\n")
        print("\n")
        print("===============================")
        print("\033[92m" + 'All Tests passed' + "\033[0m")
        print("===============================")
        print("\n")
        print("\n")

         
              

