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
        pathfile = scriptfolder + '/py4pd/' + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            cmd = f'pd -nogui -send "start-test bang" {pathfile}' 
        else:
            print('PureData Object not found')
            sys.exit()
        # os.system(cmd)
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
        # sys.exit()
        
    elif platform.system() == 'Windows':
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + '\\py4pd\\' + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            pass
        else:
            print(f'Patch {pathfile} not found')
            sys.exit()
        os.system(f'cmd /c "\"C://Program Files//Pd//bin//pd.exe\" -send \"start-test bang\" \"{pathfile}\""')
        output = subprocess.run(f'"C:\\Program Files\\Pd\\bin\\pd.exe" -nogui -send "start-test bang" "{pathfile}"', capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif platform.system() == 'Darwin':
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + '/py4pd/' + pdpatch
        # check if file exists
        if os.path.isfile(pathfile):
            pass
        else:
            print(f'Patch {pathfile} not found')
            sys.exit()

        cmd = '/Applications/Pd-*.app/Contents/Resources/bin/pd -stderr -send "start-test bang" ' + pathfile
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = "Fail"
        for line in outputLines:
            # check if there is PASS inside some line
            if "PASS" in line:
                lastLine = line
                break
    else:
        print('OS not supported')
        sys.exit()

    # if lastLine contains "PASS" then the test passed
    if "PASS" in lastLine:
        # print in green
        print("\033[92m" + ' Test with ' + pdpatch + ' passed' + "\033[0m")
        return "ok"
    else:
        print("\033[91m" + ' Test with ' + pdpatch + ' failed' + "\033[0m")

    
if __name__ == "__main__":
    # list all patches inside test folder
    patches = os.listdir('test')
    patches = [patch for patch in patches if patch.endswith('.pd')]
    patches.sort()
    for patch in patches:
        print("============= " + patch + " =============")
        runTest(patch)

         
              

