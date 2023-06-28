import pd


def png():
    pd.print("Ok")




def py4pdLoadObjects():
    pd.addobject(png, "gif", objimage="./resources/test.gif", objtype=pd.VIS)
    pd.addobject(png, "png", objimage="./resources/test.png", objtype=pd.VIS)
    pd.addobject(png, "doc", objimage="./resources/dog.png", objtype=pd.VIS)
    pd.addobject(png, "png25", objimage="./resources/25.png", objtype=pd.VIS)
    pd.addobject(png, "flower", objimage="./resources/flower.png", objtype=pd.VIS)
    pd.addobject(png, "none", objtype=pd.VIS)


