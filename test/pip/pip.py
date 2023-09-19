import pd


def testpip():
    pd.pip_install('om_py', global_install=True, detach_install=True)




def pip_setup():
    pd.add_object(testpip, "pip")
