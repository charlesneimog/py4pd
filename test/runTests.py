import os
import platform
import subprocess
import sys

errorInTest = 0


def runTest(pdpatch):
    global errorInTest
    if platform.system() == "Linux":
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "/" + pdpatch
        if os.path.isfile(pathfile):
            cmd = f'pd -nogui -batch -send "start-test bang" {pathfile}'
            print("Running: " + "\033[92m" + cmd + "\033[0m", end="\r")
        else:
            print("PureData Patch not found")
            sys.exit()
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, shell=True, timeout=60
            )
            outputLines = str(output).split("\\n")
        except subprocess.TimeoutExpired:
            print("\033[K", end="\r")
            print("\033[91m" + " Test with " + pdpatch + " failed, TIMEOUT" + "\033[0m")
            errorInTest += 1
            return
    elif platform.system() == "Windows":
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + pdpatch
        # check if pathfile has JUSTLINUX in it
        if "JUSTLINUX" in pathfile:
            print("Test not supported on Windows")
            return

        if os.path.isfile(pathfile):
            pass
        else:
            print(f"Patch {pathfile} not found")
            sys.exit()
        cmd = f'"C:\\Program Files\\Pd\\bin\\pd.exe" -nogui -batch -send "start-test bang" "{pathfile}"'
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, shell=True, timeout=60
            )
            outputLines = str(output).split("\\n")
        except subprocess.TimeoutExpired:
            print("\033[K", end="\r")
            print("\033[91m" + " Test with " + pdpatch + " failed, TIMEOUT" + "\033[0m")
            errorInTest += 1
            return
    elif platform.system() == "Darwin":
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "/" + pdpatch
        if "JUSTLINUX" in pathfile:
            print("Test not supported on MacOS")
            return

        # check if file exists
        if os.path.isfile(pathfile):
            pass
        else:
            print(f"Patch {pathfile} not found")
            sys.exit()
        cmd = (
            '/Applications/Pd-*.app/Contents/Resources/bin/pd -nogui -stderr -send "start-test bang" '
            + pathfile
        )
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, shell=True, timeout=60
            )
            outputLines = str(output).split("\\n")
        except subprocess.TimeoutExpired:
            print("\033[K", end="\r")
            print("\033[91m" + " Test with " + pdpatch + " failed, TIMEOUT" + "\033[0m")
            errorInTest += 1
            return
    else:
        print("OS not supported")
        sys.exit()

    # if lastLine contains "PASS" then the test passed
    passed = False
    for line in outputLines:
        if "PASS" in line or "Pass" in line:
            passed = True
    print("\033[K", end="\r")
    if passed:
        print("\033[92m" + " Test with " + pdpatch + " passed" + "\033[0m")
    else:
        for line in outputLines:
            print("\033[93m" + line + "\033[0m")
        print("\033[91m" + " Test with " + pdpatch + " failed" + "\033[0m")
        errorInTest += 1


if __name__ == "__main__":
    # list all patches inside test folder
    scriptFolder = os.path.dirname(os.path.abspath(__file__))
    patches = os.listdir(scriptFolder)
    patches = [patch for patch in patches if patch.endswith(".pd")]
    patches.sort()
    for patch in patches:
        runTest(patch)
    if errorInTest != 0:
        print("\033[91m" + f"{errorInTest} Test has failed" + "\033[0m")
        sys.exit(-1)
    elif errorInTest == 0:
        print("\n")
        print("==============================")
        print("\033[92m" + (" " * 7) + "All Tests passed" + "\033[0m")
        print("==============================")
        print("\n")
