import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------- Project Specifications ----------------
FS = 48000              # Sampling frequency: 48,000 samples per second
FILTER_ORDER = 20       # Required filter order
RP = 1                  # Maximum passband ripple: 1 dB
RS = 17                 # Minimum stopband attenuation: 17 dB
FP = 22000              # Passband edge frequency: 22,000 Hz


def design_filter():

    # Step 1: Pre-warp the digital passband edge for the bilinear transformation
    wc = 2 * FS * np.tan(np.pi * FP / FS)

    # Step 2: Create the analog elliptic filter prototype using the specifications
    z, p, k = signal.ellipap(FILTER_ORDER, RP, RS)

    # Step 3: Scale the analog prototype to the desired cutoff frequency
    z, p, k = signal.lp2lp_zpk(z, p, k, wo=wc)

    # Step 4: Convert the analog filter into a digital filter using bilinear transformation
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs=FS)

    # Step 5: Convert the filter into second-order sections for numerical stability
    sos = signal.zpk2sos(zd, pd, kd)

    return sos  # Return the completed digital filter


def plot_frequency_response(sos):

    # Calculate the filter response using a dense frequency grid
    frequencies, response = signal.sosfreqz(
        sos, worN=2**22, fs=FS
    )

    # Convert the filter magnitude into decibels
    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-14)
    )

    # Find the first frequency after FP where attenuation reaches 17 dB
    stop_indices = np.where(
        (frequencies > FP) & (magnitude_db <= -RS)
    )[0]

    # Stop if the stopband edge cannot be found
    if len(stop_indices) == 0:
        raise ValueError("Stopband edge could not be located.")

    # Record the calculated stopband edge frequency
    stopband_edge = frequencies[stop_indices[0]]

    # Separate the passband and stopband values for verification
    passband = magnitude_db[frequencies <= FP]
    stopband = magnitude_db[frequencies >= stopband_edge]

    # Calculate the maximum and minimum passband magnitudes
    passband_max = np.max(passband)
    passband_min = np.min(passband)

    # Calculate the difference between maximum and minimum passband values
    passband_ripple = passband_max - passband_min

    # Find the highest magnitude in the stopband
    stopband_max = np.max(stopband)

    # Calculate the actual filter order from the number of sections
    actual_order = sos.shape[0] * 2

    # Display the filter specifications and verification results
    print("\n-------- FILTER VERIFICATION --------")
    print("Actual filter order:", actual_order)
    print("Passband edge:", FP, "Hz")
    print("Calculated stopband edge:",
          f"{stopband_edge:.5f} Hz")
    print("Passband maximum:",
          f"{passband_max:.3f} dB")
    print("Passband minimum:",
          f"{passband_min:.3f} dB")
    print("Passband ripple:",
          f"{passband_ripple:.3f} dB")
    print("Maximum stopband magnitude:",
          f"{stopband_max:.3f} dB")

    print("-------------------------------------")

    # Calculate a smaller frequency grid for the full response graph
    full_freq, full_response = signal.sosfreqz(
        sos, worN=16384, fs=FS
    )

    # Convert the full frequency response into decibels
    full_db = 20 * np.log10(
        np.maximum(np.abs(full_response), 1e-14)
    )

    # Create two graphs: full response and zoomed transition region
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9)
    )

    # Add the main title to the figure
    fig.suptitle(
        "Elliptic IIR Low-Pass Filter Frequency Response",
        fontsize=16,
        fontweight="bold"
    )

    # ---------------- Full frequency response ----------------

    # Plot the filter response across the entire frequency range
    ax1.plot(
        full_freq, full_db,
        color="blue", linewidth=1.5,
        label="Filter response"
    )

    # Show the maximum allowed passband ripple
    ax1.axhline(
        -RP, color="green", linestyle="--",
        label="Passband limit (-1 dB)"
    )

    # Show the required stopband attenuation
    ax1.axhline(
        -RS, color="red", linestyle="--",
        label="Stopband limit (-17 dB)"
    )

    # Mark the passband edge at 22,000 Hz
    ax1.axvline(
        FP, color="green", linestyle=":",
        label="Passband edge (22,000 Hz)"
    )

    # Mark the calculated stopband edge
    ax1.axvline(
        stopband_edge, color="red", linestyle=":",
        label=f"Stopband edge ({stopband_edge:.3f} Hz)"
    )

    # Shade the passband region in green
    ax1.axvspan(
        0, FP, color="green", alpha=0.07
    )

    # Shade the transition region in orange
    ax1.axvspan(
        FP, stopband_edge, color="orange", alpha=0.45
    )

    # Shade the stopband region in red
    ax1.axvspan(
        stopband_edge, FS / 2,
        color="red", alpha=0.05
    )

    # Add graph labels, limits, grid, and legend
    ax1.set_title("Full Frequency Response")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlim(0, FS / 2)  # Display from 0 Hz to Nyquist frequency
    ax1.set_ylim(-30, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    # ---------------- Zoomed transition region ----------------

    # Set the frequency range to zoom in around the transition region
    zoom_start = FP - 0.2
    zoom_end = stopband_edge + 0.2

    # Select only the frequency values inside the zoomed range
    zoom_mask = (
        (frequencies >= zoom_start) &
        (frequencies <= zoom_end)
    )

    # Plot the filter response in the zoomed transition region
    ax2.plot(
        frequencies[zoom_mask],
        magnitude_db[zoom_mask],
        color="blue", linewidth=1.5,
        label="Filter response"
    )

    # Show the passband ripple limit
    ax2.axhline(
        -RP, color="green", linestyle="--",
        label="-1 dB passband limit"
    )

    # Show the stopband attenuation limit
    ax2.axhline(
        -RS, color="red", linestyle="--",
        label="-17 dB stopband limit"
    )

    # Mark the passband edge
    ax2.axvline(
        FP, color="green", linestyle=":",
        label="Passband edge"
    )

    # Mark the calculated stopband edge
    ax2.axvline(
        stopband_edge, color="red", linestyle=":",
        label="Stopband edge"
    )

    # Highlight the transition region between the two edges
    ax2.axvspan(
        FP, stopband_edge,
        color="orange", alpha=0.25,
        label="Transition region"
    )

    # Add graph labels, limits, grid, and legend
    ax2.set_title("Zoomed Transition Region")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_ylim(-25, 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    # Adjust graph spacing so labels do not overlap
    plt.tight_layout()

    # Save the frequency response graph in the graphs folder
    plt.savefig("graphs/filter_response.png", dpi=300)

    # Display the frequency response graphs
    plt.show()


# Run this section only when this file is executed directly
if __name__ == "__main__":

    # Design the elliptic IIR low-pass filter
    sos = design_filter()

    # Display the filter specifications
    print("Elliptic IIR filter designed successfully!")
    print("Sampling frequency:", FS, "Hz")
    print("Specified filter order:", FILTER_ORDER)
    print("Passband ripple specification:", RP, "dB")
    print("Stopband attenuation specification:", RS, "dB")

    # Verify the filter and generate its frequency response graphs
    plot_frequency_response(sos)
