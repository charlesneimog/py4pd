import pd

def pyPlayer():
    pd.add_to_player(100, [100, 200, 300])
    pd.add_to_player(500, 'hello from 500')
    pd.add_to_player(1000, 'hello from 1000')
    pd.add_to_player(2000, 'hello from 2000')


def py4pdLoadObjects():
    pd.add_object(pyPlayer, 'py.player', pyout=True, playable=True)

