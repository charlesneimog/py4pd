We have three objects to use.

- [Audio In](#audio-in)
- [Audio Out](#audio-out)
- [Audio In/Out](#audio-inout)

### Audio In

Used to output analised data from audio. Objects with Partial Trackings, Audio Descriptors, and others.

To create Audio In objects, you have two options.

```py

pd.add_object(myfft, "myfft~", objtype=pd.AUDIOIN)

```

or for more complex and complete objects

```py

myfftobj = pd.new_object("myfft~")
myfftobj.type = pd.AUDIOIN
myfftobj.addmethod_audioin(myfft)
myfftobj.addmethod_float(myfloatmethod)

```

!!! Warning

    The *first inlet* of this object always needs to be **audio**, in Python, the first also always will be audio. For audio `py4pd` uses a numpy array, you can use `snake~ in` to create objects with multiple audio inputs, each channel will be one dimension of the array.

<p align="center">
    <img src="../../../examples/audioin/audioin.png" width="50%" alt="Audio Input Image" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
</p>

??? example end "Python Code"

    ```py

    import pd
    import numpy

    def audioin(audio):
        fft = numpy.fft.fft(audio)
        fft = numpy.real(fft)
        return fft.tolist() # numpy can just be outputed when pyout=True

    def libraryname_setup():
        pd.add_object(audioin, "audioin", objtype=pd.AUDIOIN)

    ```

### Audio Out

Used to create audio using Python. Objects that create sinusoids, some special noise, and others.

To create an Audio out object, in `pd.add_object` we add the `objtype=pd.AUDIOOUT`.

The audio function always needs to return a numpy array.

<p align="center">
    <img src="../../../examples/audioout/audioout.png" width="35%" alt="Audio Output Image" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
</p>

### Audio (in/out)

Used to manipulate Audio. FFT, reverbs, and others.

To create an Audio object (audio input and output), in `pd.add_object` we add the `objtype=pd.AUDIO`.

<p align="center">
    <img src="../../../examples/audio/audio.png" width="35%" alt="Audio Image" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
</p>

??? example end "Python Code"

    ```python

        import pd
        import numpy

        def audio(audio, amplitude):
            if amplitude is None:
                amplitude = 0.2
            audio = numpy.multiply(audio, amplitude)
            return audio


        def libraryname_setup():
            pd.add_object(audio, "audio", objtype=pd.AUDIO)

    ```
