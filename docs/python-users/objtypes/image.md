Used to create functions to show something. Like Scores, Audio descriptors, and others.

To create vis object, in `pd.add_object` we add the `objtype=pd.VIS`. Inside the function, we always need the `pd.show_image` method, without it, anything will be showed.
For `pd.VIS` objects, we have some options in `pd.add_object`.

See the example:

??? example end "Python Code"

    ``` py

    import pd
    import audioflux as af
    import matplotlib.pyplot as plt
    from audioflux.display import fill_plot, fill_wave
    from audioflux.type import SpectralFilterBankScaleType, SpectralDataType
    import numpy as np

    def descriptors():
        audio_arr, sr = af.read(pd.get_home_folder() + "/Hp-ord-A4-mf-N-N.wav")
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
        tempfile = pd.get_temp_dir() + "/descritores.png"
        plt.savefig(tempfile)
        pd.show_image(tempfile)
        pd.print("Data plotted")

    def libraryname_setup():
        pd.add_object(descriptors, "descritores", objtype=pd.VIS, figsize=(640, 480))

    ```

<p align="center">
    <img src="../../../../examples/descriptors/descriptors.png" width="40%" alt="Descriptors Image" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
</p>

## py4pd-score

With `py4pd-score` you can use tradicional scores inside PureData patches. It is highly inspired in OpenMusic score, when `py4pd` be very stable, I want to implement the `voice`, `note` objects, for now just chord is implemented.

<p align="center">
    <img src="../../../../examples/score/score.gif" width="50%" alt="Scores" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
</p>
