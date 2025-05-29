import pd
def logpost():
    pd.logpost(1, "Level 1")
    pd.logpost(2, "Level 2")
    pd.logpost(3, "Level 3")
    pd.logpost(4, "Level 4")


def logpost_setup():
    pd.add_object(logpost, "py.logs")
