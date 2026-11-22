import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

from filter_design import design_filter, FS


# -------------------- File Paths --------------------

AUDIO_PATH = "audio/spongebob_voice.wav"
NOISY_PATH = "audio/spongebob_noisy.wav"
OUTPUT_PATH = "audio/spongebob_filtered.wav"


# -------------------- Step 1: Load Original Audio --------------------

clean_audio, sample_rate = sf.read(AUDIO_PATH)

print("\n========== ORIGINAL AUDIO ==========")
print("Sampling frequency:", sample_rate, "Hz")
print("Number of samples:", len(clean_audio))
print("Audio duration:", len(clean_audio) / sample_rate, "seconds")
print("====================================")


# Verify sampling frequency
if sample_rate != FS:
    raise ValueError(
        "Audio sampling frequency does not match filter sampling frequency!"
    )


# -------------------- Step 2: Add High-Frequency Noise --------------------
# CHANGED: the original recording has no noise (nothing above ~10.5 kHz),
# so a 22 kHz low-pass filter has nothing to remove. We add noise in the
# 22.5 kHz - 24 kHz range, which is the only range this filter can remove
# (Nyquist = 24 kHz).

rng = np.random.default_rng(3313)                       # same noise every run
white_noise = rng.standard_normal(len(clean_audio))
highpass = signal.butter(8, 22500, btype="highpass", fs=FS, output="sos")
noise = signal.sosfilt(highpass, white_noise)
noise = 0.05 * noise / np.std(noise)                    # noise strength

audio = clean_audio + noise                             # noisy input signal

sf.write(NOISY_PATH, audio, sample_rate)


# -------------------- Step 3: Design Elliptic Filter --------------------

print("\nDesigning elliptic IIR low-pass filter...")

sos = design_filter()

print("Filter designed successfully!")


# -------------------- Step 4: Apply Filter --------------------
# CHANGED: sosfilt instead of sosfiltfilt. sosfiltfilt runs the filter
# forwards AND backwards, so the signal sees |H|^2 (-2 dB in the passband
# and double attenuation), which is not the filter we designed.

print("\nApplying filter to noisy audio...")

filtered_audio = signal.sosfilt(sos, audio)

print("Filtering completed!")


# -------------------- Step 5: Save Filtered Audio --------------------

sf.write(OUTPUT_PATH, filtered_audio, sample_rate)

print("\nFiltered audio saved successfully!")
print("Output file:", OUTPUT_PATH)


# -------------------- Step 6: Plot Audio Signals --------------------

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, audio, color="blue", linewidth=0.7)
plt.title("Audio Before Filtering (voice + high-frequency noise)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.ylim(-0.4, 0.4)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(time, filtered_audio, color="red", linewidth=0.7)
plt.title("Audio After Filtering")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.ylim(-0.4, 0.4)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graphs/audio_comparison.png", dpi=300)
plt.show()


# -------------------- Step 7: Calculate DFT --------------------

print("\nCalculating DFT of noisy and filtered audio...")

# CHANGED: two-sided DFT (-24 kHz to +24 kHz) like the project example
original_fft = np.fft.fftshift(np.fft.fft(audio))
filtered_fft = np.fft.fftshift(np.fft.fft(filtered_audio))

frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(audio), d=1 / sample_rate)
)

# Normalize both using the noisy audio's maximum
normalization_factor = np.max(np.abs(original_fft))
original_normalized = np.abs(original_fft) / normalization_factor
filtered_normalized = np.abs(filtered_fft) / normalization_factor

# CHANGED: filter magnitude response on the same frequency axis (orange line)
_, H = signal.sosfreqz(sos, worN=2 * np.pi * frequencies / FS)
filter_magnitude = np.abs(H)

print("DFT calculation completed!")


# -------------------- Step 8: Plot DFT Comparison --------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(frequencies, original_normalized, color="blue", linewidth=0.8)
plt.plot(frequencies, filter_magnitude, color="orange", linewidth=1.5,
         label="Filter response")
plt.title("DFT Before Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.xlim(-sample_rate / 2, sample_rate / 2)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.plot(frequencies, filtered_normalized, color="red", linewidth=0.8)
plt.plot(frequencies, filter_magnitude, color="orange", linewidth=1.5,
         label="Filter response")
plt.title("DFT After Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.xlim(-sample_rate / 2, sample_rate / 2)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("graphs/dft_comparison.png", dpi=300)
plt.show()


# -------------------- Program Complete --------------------

print("\n========== PROJECT COMPLETE ==========")
print("Noisy audio created and saved.")
print("Elliptic IIR filter applied successfully.")
print("Filtered audio saved successfully.")
print("Audio signal graphs generated.")
print("DFT comparison graph generated.")
print("======================================")
