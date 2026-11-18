import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# TEAM 6: ELLIPTIC IIR LOW-PASS FILTER
# DESIGN METHOD: BILINEAR TRANSFORMATION
# ==========================================

# 1. Filter specifications
Fs = 48000       # Sampling frequency (Hz)
Fp = 22000       # Passband edge (Hz)
Rp = 1            # Passband ripple (dB)
Rs = 40           # Temporary stopband attenuation (dB)
N = 20            # Filter order

# 2. Prewarp the digital passband frequency
# This compensates for frequency warping
# caused by the bilinear transformation.

Wp = 2 * Fs * np.tan(np.pi * Fp / Fs)

# 3. Create normalized analog elliptic prototype
z, p, k = signal.ellipap(N, Rp, Rs)

# 4. Scale the analog filter to the desired frequency
z, p, k = signal.lp2lp_zpk(z, p, k, wo=Wp)

# 5. Convert analog filter into digital filter
# using the bilinear transformation.

zd, pd, kd = signal.bilinear_zpk(z, p, k, fs=Fs)

# 6. Convert to second-order sections
# for numerical stability.

sos = signal.zpk2sos(zd, pd, kd)

# 7. Calculate frequency response
frequencies, response = signal.sosfreqz(
    sos, worN=8192, fs=Fs
)

magnitude_db = 20 * np.log10(
    np.maximum(np.abs(response), 1e-12)
)

# 8. Plot frequency response
plt.figure(figsize=(10, 6))

plt.plot(frequencies, magnitude_db)

plt.title("Team 6 - Elliptic IIR Low-pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

# Mark important filter frequencies
plt.axvline(Fp, color='red', linestyle='--',
            label='Passband Edge (22 kHz)')

plt.axhline(-Rp, color='green', linestyle=':',
            label='Passband Ripple (-1 dB)')

plt.axhline(-Rs, color='purple', linestyle=':',
            label='Stopband Attenuation (-40 dB)')

plt.grid(True)
plt.xlim(0, Fs / 2)
plt.ylim(-80, 5)
plt.legend()

plt.tight_layout()
plt.savefig("graphs/filter_response.png")
plt.show()

print("Filter design completed!")
print("Filter order:", N)
print("Sampling frequency:", Fs, "Hz")
print("Passband ripple:", Rp, "dB")
print("Stopband attenuation:", Rs, "dB")
