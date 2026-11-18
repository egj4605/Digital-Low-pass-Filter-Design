import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# TEAM 6 - AUDIO PROCESSING
# ==========================================

# 1. Audio specifications
Fs = 48000

# 2. Load original voice recording
audio, sample_rate = sf.read("audio/spongebob_voice.wav")

# Convert to mono if necessary
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

print("Original audio loaded!")
print("Sampling rate:", sample_rate)
print("Duration:", len(audio) / sample_rate, "seconds")

# 3. Generate high-frequency noise
np.random.seed(42)

time = np.arange(len(audio)) / Fs

noise = np.random.normal(0, 0.05, len(audio))

# Focus added noise above 22 kHz
sos_noise = signal.butter(
    4, 22000, btype='highpass',
    fs=Fs, output='sos'
)

high_frequency_noise = signal.sosfilt(
    sos_noise, noise
)

# Add noise to original recording
noisy_audio = audio + high_frequency_noise

# Normalize to avoid clipping
noisy_audio = noisy_audio / max(
    np.max(np.abs(noisy_audio)), 1e-12
) * 0.95

# 4. Save noisy recording
sf.write(
    "audio/spongebob_noisy.wav",
    noisy_audio,
    Fs
)

print("Noisy audio saved!")

# 5. Design Team 6 elliptic filter
N = 20
Rp = 1
Rs = 40

Fp = 22000

Wp = 2 * Fs * np.tan(np.pi * Fp / Fs)

z, p, k = signal.ellipap(N, Rp, Rs)

z, p, k = signal.lp2lp_zpk(
    z, p, k, wo=Wp
)

zd, pd, kd = signal.bilinear_zpk(
    z, p, k, fs=Fs
)

sos = signal.zpk2sos(zd, pd, kd)

# 6. Apply filter
filtered_audio = signal.sosfiltfilt(
    sos, noisy_audio
)

# Save filtered recording
sf.write(
    "audio/spongebob_filtered.wav",
    filtered_audio,
    Fs
)

print("Filtered audio saved!")

# 7. Plot original, noisy, and filtered audio
plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(time, audio)
plt.title("Original SpongeBob Audio")

plt.subplot(3, 1, 2)
plt.plot(time, noisy_audio)
plt.title("Noisy SpongeBob Audio")

plt.subplot(3, 1, 3)
plt.plot(time, filtered_audio)
plt.title("Filtered SpongeBob Audio")

plt.xlabel("Time (seconds)")
plt.tight_layout()

plt.savefig("graphs/audio_comparison.png")
plt.show()

print("Audio processing completed!")
