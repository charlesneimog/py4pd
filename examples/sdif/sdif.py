import pd

try:
    import loristrck    
except ImportError as e:
    pd.error("LorisTrck not found. Trying to install")
    try:
        pd.pipinstall("local", "loristrck")
        import loristrck
    except Exception as e:
        pd.error("LorisTrck not found. Please install it manually")




def readSdifFromSound(thing):
    method = thing[0]
    filename = thing[1]

    if method == "tabread":
        samples = pd.tabread(filename, numpy=True)
        sr = pd.samplerate()
        partials = loristrck.analyze(samples, sr, resolution=60)
        return partials


    elif method == "readfile":
        samples, sr = loristrck.util.sndreadmono("/path/to/sndfile.wav")






def py4pdLoadObjects():
    pd.addobject(readSdifFromSound, "sdif.read", pyout=True)
