import pd
import librosa
import matplotlib.pyplot as plt
import numpy as np


def sonogram(tabname):
    sr = pd.samplerate()
    y = np.array(pd.tabread(tabname))

    pd.tabread(tabname)
    
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
