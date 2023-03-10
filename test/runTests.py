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
        pathfile = scriptfolder + '/py4pd_Linux/' + pdpatch
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
        pathfile = scriptfolder + '\\py4pd_WIN64\\' + pdpatch
        print(f'"C:\\Program Files\\Pd\\bin\\pd.exe" -send "start-test bang" "{pathfile}"')
        os.system(f'cmd /c "\"C://Program Files//Pd//bin//pd.exe\" -send \"start-test bang\" \"{pathfile}\""')
        output = subprocess.run(f'"C:\\Program Files\\Pd\\bin\\pd.exe" -nogui -send "start-test bang" "{pathfile}"', capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif platform.system() == 'Darwin':
        cmd = '/Applications/Pd-*.app/Contents/Resources/bin/pd -stderr -send "start-test bang" py4pd_macOS-Intel/' + pdpatch
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    else:
        print('OS not supported')
        sys.exit()

    # if lastLine contains "PASS" then the test passed
    if "PASS" in lastLine:
        # print in green
        print("\033[92m" + ' Test with ' + pdpatch + ' passed' + "\033[0m")
        return "ok"
    else:
        for line in outputLines:
            errorInTest += 1

            if "FAIL" in line:
                print("\033[91m" + line + "\033[0m")
            # if the line contains "PASS" then print in green
            elif "PASS" in line:
                print("\033[92m" + line + "\033[0m")
            # otherwise print normally
            else:
                print(line)
        # sys.exit(1)
    
if __name__ == "__main__":
    patches = ['01-simpleRun.pd', '02-noArgs.pd', '03-sequencialRunning.pd', '04-neoscore.pd', '05-wrongNumberOfArgs.pd', '06-setDifferentFunctions.pd', '07-audioin-Neoscore.pd']
    for patch in patches:
        print("=============" + patch + "=============")
        runTest(patch)
    if errorInTest > 0:
        print("\033[91m" + f'{errorInTest} tests failed' + "\033[0m")
        sys.exit(1)
    else:
        print("\033[92m" + 'All tests passed' + "\033[0m")
        sys.exit(0)


         
              

