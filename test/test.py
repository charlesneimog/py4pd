import os
import sys

if sys.platform == "win32":
    os.chdir("../py4pd_WIN32")
    os.system('"C:\\Program Files\\Pure Data\\bin\\pd.exe" -nogui test.pd')
    
elif sys.platform == "darwin":
    os.chdir("../py4pd_MACIntel")
    os.system("/Applications/Pd-*.app/Contents/Resources/bin/pd -nogui test/test.pd")
    
elif sys.platform == "linux2":
    os.chdir("../py4pd_LINUX")
    os.system("pd -nogui test/test.pd")
              

