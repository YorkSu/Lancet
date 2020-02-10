import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


filepath = './dataset/york/1_Mix.wav'
y, sr = librosa.load(filepath, sr=None)

d = librosa.get_duration(y=y, sr=sr)
z = librosa.zero_crossings(y)
zp = sum(z) / len(z)
print("sample:", len(y))
print("second:", d)
print("zero crossing:", zp)

### raw signal
plt.plot(list(range(len(y))), y)
plt.title('raw signal')
plt.show()

### chroma_stft
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chroma_stft, y_axis='chroma')
plt.colorbar(format='%+2.0f dB')
plt.title('chroma_stft')
plt.show()

### chroma_cqt
chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('chroma_cqt')
plt.show()

### chroma_cens
chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('chroma_cens')
plt.show()

### melspectrogram
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)#, n_fft=1024, hop_length=512, n_mels=512
plt.imshow(melspectrogram, cmap='viridis')
plt.gca().invert_yaxis()
plt.title('melspectrogram')
plt.show()

### mfcc
mfcc = librosa.feature.mfcc(y=y, sr=sr)#, n_mfcc=1000, n_fft=1024, hop_length=512
librosa.display.specshow(mfcc, y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('mfcc')
plt.show()

### rms
rms = librosa.feature.rms(y=y)
rms = np.ravel(rms)
plt.plot(list(range(len(rms))), rms)
plt.title('rms')
plt.show()

### spectral_centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_centroid = np.ravel(spectral_centroid)
plt.plot(list(range(len(spectral_centroid))), spectral_centroid)
plt.title('spectral_centroid')
plt.show()

### spectral_bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spectral_bandwidth = np.ravel(spectral_bandwidth)
plt.plot(list(range(len(spectral_bandwidth))), spectral_bandwidth)
plt.title('spectral_bandwidth')
plt.show()

### spectral_contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
librosa.display.specshow(spectral_contrast, y_axis='linear', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('spectral_contrast')
plt.show()

### spectral_flatness
spectral_flatness = librosa.feature.spectral_flatness(y=y)
spectral_flatness = np.ravel(spectral_flatness)
plt.plot(list(range(len(spectral_flatness))), spectral_flatness)
plt.title('spectral_flatness')
plt.show()

### spectral_rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
spectral_rolloff = np.ravel(spectral_rolloff)
plt.plot(list(range(len(spectral_rolloff))), spectral_rolloff)
plt.title('spectral_rolloff')
plt.show()

### tonnetz
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
librosa.display.specshow(tonnetz, y_axis='linear', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('tonnetz')
plt.show()

### tempogram
tempogram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(tempogram, y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('tempogram')
plt.show()


