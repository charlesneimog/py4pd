# import os
# import sys
# import requests

# version = '0.5.0'
# response = requests.get("https://api.github.com/repos/charlesneimog/py4pd/releases/latest")
# objectVersion = response.json()['tag_name']
# repo = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

# if version != objectVersion and repo == 'master':
#     print('Version mismatch. Please update the object version in the uploadobject.py file.')

def runTest():
    import os
    import subprocess
    import sys
    if os.name == 'posix':
        cmd = 'pd -nogui -send "start-test bang"  py4pd_Linux/test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif os.name == 'nt':
        cmd = 'cmd /c "C:\Program Files\Pd\bin\pd.exe" -send start-test bang -nogui py4pd_WIN64\test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif os.name == 'mac':
        cmd = '/Applications/Pd-*.app/Contents/Resources/bin/pd -nogui -send "start-test bang" py4pd_macOS-Intel/test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    # if lastLine contains "PASS" then the test passed
    if "PASS" in lastLine:
        # print in green
        print("\033[92m" + ' ALL TESTS PASSED ' + "\033[0m")
        return "ok"
    else:
        # split all the lines
        for line in outputLines:
            # if the line contains "FAIL" then print in red
            if "FAIL" in line:
                print("\033[91m" + line + "\033[0m")
            # if the line contains "PASS" then print in green
            elif "PASS" in line:
                print("\033[92m" + line + "\033[0m")
            # otherwise print normally
            else:
                print(line)
        sys.exit(1)
    
    
    
         
              

