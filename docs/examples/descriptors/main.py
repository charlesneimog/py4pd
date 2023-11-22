import pd
import audioflux as af
import matplotlib.pyplot as plt
from audioflux.display import fill_plot, fill_wave
from audioflux.type import SpectralFilterBankScaleType, SpectralDataType
import numpy as np

def descriptors():
    audio_arr, sr = af.read(pd.get_patch_dir() + "/flute.wav")
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

def py4pdLoadObjects():
    pd.add_object(descriptors, "descritores", objtype=pd.VIS, figsize=(640, 480))

