import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

from filter_design import design_filter, FS


# -------------------- File Paths --------------------

AUDIO_PATH = "audio/spongebob_voice.wav"      # input 
OUTPUT_PATH = "audio/spongebob_filtered.wav"  # output


# -------------------- Step 1: LOAD THE ORIGINAL AUDIO --------------------

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


# -------------------- Step 2: SET THE ORIGINAL INPUT AUDIO --------------------

audio = clean_audio


# -------------------- Step 3: DESIGN THE ELLIPTIC FILTER --------------------

print("\nDesigning elliptic IIR low-pass filter...")

sos = design_filter()

print("Filter designed successfully!")


# -------------------- Step 4: APPLY THE FILTER TO THE ORIGINAL AUDIO --------------------


print("\nApplying filter to original audio...")

filtered_audio = signal.sosfilt(sos, audio)

print("Filtering completed!")


# -------------------- Step 5: SAVE THE FILTERED AUDIO --------------------

sf.write(OUTPUT_PATH, filtered_audio, sample_rate)

print("\nFiltered audio saved successfully!")
print("Output file:", OUTPUT_PATH)


# -------------------- Step 6: PLOT THE AUDIO WAVEFORMS --------------------

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, audio, color="blue", linewidth=0.7)
plt.title("Audio Before Filtering")
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
plt.savefig("graphs/audio_signals.png", dpi=300)
plt.show()


# -------------------- Step 7: CALCULATE THE DISCRETE FOURIER TRANSFORM (DFT) --------------------

print("\nCalculating DFT of original and filtered audio...")

# calculate two-sided DFT (-24 kHz to +24 kHz)
original_fft = np.fft.fftshift(np.fft.fft(audio))
filtered_fft = np.fft.fftshift(np.fft.fft(filtered_audio))

frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(audio), d=1 / sample_rate)
)

# -------------------- Step 8: NORMALIZE THE DFT MAGNITUDES ----------------
normalization_factor = np.max(np.abs(original_fft))

original_normalized = np.abs(original_fft) / normalization_factor
filtered_normalized = np.abs(filtered_fft) / normalization_factor


# -------------------- Step 9: NORMALIZE THE DFT MAGNITUDES ------------------
_, H = signal.sosfreqz(sos, worN=2 * np.pi * frequencies / FS)
filter_magnitude = np.abs(H)

print("DFT calculation completed!")


# -------------------- Step 10: PLOT THE DFT COMPARISON --------------------

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


# -------------------- Step 11: DISPLAY PROJECT COMPLETION MESSAGE --------------------

print("\n========== PROJECT COMPLETE ==========")
print("Original audio processed successfully.")
print("Elliptic IIR filter applied successfully.")
print("Filtered audio saved successfully.")
print("Audio signal graphs generated.")
print("DFT comparison graph generated.")
print("======================================")