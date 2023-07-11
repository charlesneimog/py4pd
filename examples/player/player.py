import pd

def pyPlayer():
    pd.add2player(100, [100, 200, 300])
    pd.add2player(500, 'hello from 500')
    pd.add2player(1000, 'hello from 1000')
    pd.add2player(2000, 'hello from 2000')


def py4pdLoadObjects():
    pd.addobject(pyPlayer, 'py.player', helppatch="myplayer", pyout=True, playable=True)

