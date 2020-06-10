import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('website')

files = ['bass', 'mix', 'other', 'voice', 'drums']
# mpl.rcParams['savefig.facecolor']

for file in files:
    wave, _ = librosa.load(f"{file}.wav")

    plt.figure()
    plt.axis('off')
    librosa.display.waveplot(wave, color="#74a25d")
    plt.savefig(f"{file}_wav.png")
