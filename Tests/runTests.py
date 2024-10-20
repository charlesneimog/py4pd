import argparse
import os
import platform
import subprocess
import sys

errorInTest = 0


def runTest(pdpatch):
    global errorInTest

    # the pdpatch start with a number divided by -, get this number
    pdpatchNumber = int(pdpatch.split("-")[0])
    if pdpatchNumber == 10 or pdpatchNumber == 70 or pdpatchNumber == 71:
        timeout = 120
    else:
        timeout = 15

    if platform.system() == "Linux":
        thisSCRIPT = os.path.abspath(__file__)
        thisFOLDER = os.path.dirname(thisSCRIPT)
        completPathPatch = thisFOLDER + "/" + pdpatch
        os.chdir(thisFOLDER)
        if os.path.isfile(pdpatch):
            cmd = f'pd -nogui -send "start-test bang" "{completPathPatch}"'
        else:
            print("PureData Patch not found")
            sys.exit()
        try:
            output = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,
                timeout=timeout,
            )
            outputLines = str(output).split("\\n")
        except subprocess.TimeoutExpired:
            print("Test with " + pdpatch + " failed, TIMEOUT")
            errorInTest += 1
            return
    elif platform.system() == "Windows":
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "\\" + pdpatch
        if os.path.isfile(pathfile):
            pass
        else:
            print(f"Patch {pathfile} not found")
            sys.exit()
        cmd = [
            "..\\pd\\bin\\pd.exe",
            "-nogui",
            "-batch",
            "-send",
            "start-test bang",
            pathfile,
        ]
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                universal_newlines=True,
                timeout=timeout,
            )
            outputLines = []
            stderrTOKENS = str(process.stderr).split("\n")
            for line in stderrTOKENS:
                if "error:" in line:
                    outputLines.append(line.replace("error:", ""))
                else:
                    outputLines.append(line)

        except subprocess.TimeoutExpired:
            print("\033[K", end="\r")
            print("Test with " + pdpatch + " failed")
            errorInTest += 1
            return
    elif platform.system() == "Darwin":
        scriptfile = os.path.abspath(__file__)
        scriptfolder = os.path.dirname(scriptfile)
        pathfile = scriptfolder + "/" + pdpatch

        # check if file exists
        if not os.path.isfile(pathfile):
            print(f"Patch {pathfile} not found")
            sys.exit()
        cmd = (
            f'/Applications/Pd-*.app/Contents/Resources/bin/pd -stderr -nogui -batch -send "start-test bang" '
            + '"'
            + pathfile
            + '"'
        )
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, shell=True, timeout=timeout
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

    try:
        try:
            if passed:
                print("\033[92m" + " ✅️ Test with " + pdpatch + " passed" + "\033[0m")
            else:
                print("\033[91m" + " ❌️ Test with " + pdpatch + " failed" + "\033[0m\n")
                for line in outputLines:
                    print("\033[93m" + line + "\033[0m")
                errorInTest += 1
        except Exception as e:
            sys.stdout.reconfigure(encoding="utf-8")
            if passed:
                print(f" ✅️ Test with {pdpatch} passed")
            else:
                print(f" ❌️ Test with {pdpatch} failed")
                for line in outputLines:
                    print(line)
                errorInTest += 1
    except Exception as e:
        print(e)
        if passed:
            print(f" Test with {pdpatch} passed")
        else:
            print(f" Test with {pdpatch} failed")
            for line in outputLines:
                print(line)
            errorInTest += 1


if __name__ == "__main__":
    # list all patches inside test folder
    args = argparse.ArgumentParser()
    # create an argument where I can set -rt 6 7 11, then we will run the pates with the given rt values
    args.add_argument("-tn", type=str, nargs="+", required=False)
    args = args.parse_args()
    testNumbers = args.tn

    scriptFolder = os.path.dirname(os.path.abspath(__file__))
    patches = os.listdir(scriptFolder)
    patches = [patch for patch in patches if patch.endswith(".pd")]
    patches.sort()
    for patch in patches:
        # get patch file name
        patchName = os.path.basename(patch)
        patchNumber = int(patchName.split("-")[0])
        if testNumbers:
            for testNumber in testNumbers:
                if patchNumber == int(testNumber):
                    runTest(patch)
        else:
            runTest(patch)
        # get the test number

    if errorInTest != 0:
        print("\n")
        print("==============================")
        print("\033[91m" + (" " * 7) + f"{errorInTest} Test has failed" + "\033[0m")
        print("==============================")
        print("\n")
        sys.exit(-1)
    elif errorInTest == 0:
        print("\n")
        print("==============================")
        print("\033[92m" + (" " * 7) + "All Tests passed" + "\033[0m")
        print("==============================")
        print("\n")
