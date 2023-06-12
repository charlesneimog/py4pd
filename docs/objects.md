# Pd Objects with Python

With the `py4pd` it is possible to create new PureData objects using Python. For that, you need to declare your Python functions and then create a function called `py4pdLoadObjects`. Inside this function we use `pd.addobject` (check the [docs](https://www.charlesneimog.com/py4pd/python-users/#pdaddobject)) to add all functions that you want to use as objects.


See the Python Code:

``` py title="myNewPdObjects.py"

import pd

def mysumObject(a, b, c, d):
    return a + b + c + d

def py4pdLoadObjects():
    pd.addobject(mysumObject, "mysumObject") # function, string with name of the object
    
    # My License, Name and University, others information
    pd.print("", show_prefix=False)
    pd.print("GPL3 | by Charles K. Neimog", show_prefix=False)
    pd.print("University of São Paulo", show_prefix=False)
    pd.print("", show_prefix=False)

```

In the code above, we create a new object called `mysymObject`. It is saved inside an script called `myNewPdObjects.py`. To load this script in PureData how need to follow these steps:

* Copy the script `myNewPdObjects.py` for the folder where your PureData patch is.
* Create a new `py4pd` with this config: `py4pd -lib myNewPdObjects`.
* Create the new object, in this case `mysumObject`.

Following this steps we have this patch:

<p align="center">
    <img src="../examples/createobj/mynewpdobject.png" width="50%"</img>
</p>

### <h3 style="text-align:center"> **Types of Objects** </h3>

In the example above we create ordinary objects. With `py4pd` we can create 5 types of objects: 

1. Ordinary Objects `=>` Used to create functions like sum numbers, convertion between datas (svg to scores, for example), etc.
2. Visualization Objects `=>` Used to create functions to show something. Like Scores, Audio descriptors, and others.
3. Audio In Objects `=>` Used to output analised data from audio. Objects with Partial Trackings, Audio Descriptors, and others.
4. Audio Out Objects `=>` Used to create audio using Python. Objects that creates sinusoids, some special noise and others.
5. Audio (in and out) Objects `=>` Used to manipulations of Audio. FFT, reverbs, and others. 

<br> 

#### <h4 style="text-align:center">Visual Object</h4>

To create vis object, in `pd.addobject` we add the `objtype=pd.VIS`. Inside the function, we always need the `pd.show` method, without it, anything will be showed. 
For `pd.VIS` objects, we have some options in `pd.addobject`.

* `figsize`: It set the size of the figure that will be showed, this is more for aesthetic reasons (the figure will always be resized).

See the example:

??? example end "Python Code"
    ```py 

    import pd
    import audioflux as af
    import matplotlib.pyplot as plt
    from audioflux.display import fill_plot, fill_wave
    from audioflux.type import SpectralFilterBankScaleType, SpectralDataType
    import numpy as np

    def descriptors():
        audio_arr, sr = af.read(pd.home() + "/Hp-ord-A4-mf-N-N.wav")
        bft_obj = af.BFT(num=2049, samplate=sr, radix2_exp=12, slide_length=1024,
                       data_type=SpectralDataType.MAG,
                       scale_type=SpectralFilterBankScaleType.LINEAR)
        spec_arr = bft_obj.bft(audio_arr)
        spec_arr = np.abs(spec_arr)
        spectral_obj = af.Spectral(num=bft_obj.num,
                                   fre_band_arr=bft_obj.get_fre_band_arr())
        n_time = spec_arr.shape[-1]  # Or use bft_obj.cal_time_length(audio_arr.shape[-1])
        spectral_obj.set_time_length(n_time)
        hfc_arr = spectral_obj.hfc(spec_arr)
        cen_arr = spectral_obj.centroid(spec_arr) 

        fig, ax = plt.subplots(nrows=3, sharex=True)
        fill_wave(audio_arr, samplate=sr, axes=ax[0])
        times = np.arange(0, len(hfc_arr)) * (bft_obj.slide_length / bft_obj.samplate)
        fill_plot(times, hfc_arr, axes=ax[1], label='hfc')
        fill_plot(times, cen_arr, axes=ax[2], label="Centroid")
        tempfile = pd.tempfolder() + "/descritores.png"
        plt.savefig(tempfile)
        pd.show(tempfile)
        pd.print("Data plotted")

    def py4pdLoadObjects():
        pd.addobject(descriptors, "descritores", objtype=pd.VIS, figsize=(640, 480))
    ```
    
<p align="center">
    <img src="../examples/descriptors/descriptors.png" width="50%"</img>
</p>

<br>
--------------------------------------
#### <h4 style="text-align:center"> Audio In Object </h4>

To create Audio In object, in `pd.addobject` we add the `objtype=pd.AUDIOIN`. 

!!! warning "The first inlet of this objects always need to be audio"

<p align="center">
    <img src="../examples/audioin/audioin.png" width="50%"</img>
</p>

??? example end "Python Code"
    ```py 

    import pd
    import numpy
    
    def audioin(audio):
        fft = numpy.fft.fft(audio)
        fft = numpy.real(fft) 
        return fft.tolist() # numpy can just be outputed when pyout=True

    def py4pdLoadObjects():
        pd.addobject(audioin, "audioin", objtype=pd.AUDIOIN)
    ```
 
--------------------------------------
#### <h4 style="text-align:center"> Audio Out Object </h4>

To create Audio out object, in `pd.addobject` we add the `objtype=pd.AUDIOOUT`. 

<p align="center">
    <img src="../examples/audioout/audioout.png" width="50%"</img>
</p>

??? example end "Python Code"
    ```py 

    import pd
    import numpy
    
    def audioin(audio):
        fft = numpy.fft.fft(audio)
        fft = numpy.real(fft) 
        return fft.tolist() # numpy can just be outputed when pyout=True

    def py4pdLoadObjects():
        pd.addobject(audioin, "audioin", objtype=pd.AUDIOIN)
    ```

--------------------------------------
#### <h4 style="text-align:center"> Audio Object </h4>

To create Audio object (audio input and output), in `pd.addobject` we add the `objtype=pd.AUDIO`. 

<p align="center">
    <img src="../examples/audio/audio.png" width="50%"</img>
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


    def py4pdLoadObjects():
        pd.addobject(audio, "audio", objtype=pd.AUDIO)
        
    ```
