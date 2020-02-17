import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


filepath = './dataset/york/1_Mix.wav'

y, sr = librosa.load(filepath, sr=None)

### stft-spectrogram (*1025, t) *1+n_fft/2
stft = librosa.stft(y=y, n_fft=2048)
stft = np.abs(stft)
stft = stft.astype(np.float32)
print(stft.shape, stft.dtype)
d_stft = librosa.amplitude_to_db(stft, ref=np.max)
librosa.display.specshow(d_stft, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('stft-spectrogram')
plt.show()

### cqt-spectrogram (*84, t) *n_bins
cqt = librosa.cqt(y, sr)
cqt = np.abs(cqt)
cqt = cqt.astype(np.float32)
print(cqt.shape, cqt.dtype)
d_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
librosa.display.specshow(d_cqt, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('cqt-spectrogram')
plt.show()

### hybrid-cqt-spectrogram (混合常量Q变换) (*84, t) *n_bins
hcqt = librosa.hybrid_cqt(y=y, sr=sr)
hcqt = np.abs(hcqt)
hcqt = hcqt.astype(np.float32)
print(hcqt.shape, hcqt.dtype)
d_hcqt = librosa.amplitude_to_db(hcqt, ref=np.max)
librosa.display.specshow(d_hcqt, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('hybrid-cqt-spectrogram')
plt.show()

### pseudo-cqt-spectrogram (伪常量Q变换) (*84, t) *n_bins
pcqt = librosa.pseudo_cqt(y=y, sr=sr)
pcqt = np.abs(pcqt)
pcqt = pcqt.astype(np.float32)
print(pcqt.shape, pcqt.dtype)
d_pcqt = librosa.amplitude_to_db(pcqt, ref=np.max)
librosa.display.specshow(d_pcqt, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('hybrid-cqt-spectrogram')
plt.show()

### mel-spectrogram (*128, t) *n_mels
mel = librosa.feature.melspectrogram(y=y, sr=sr)#, n_mels=256
print(mel.shape, mel.dtype)
d_mel = librosa.power_to_db(mel, ref=np.max)
librosa.display.specshow(d_mel, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('mel-spectrogram')
plt.show()



