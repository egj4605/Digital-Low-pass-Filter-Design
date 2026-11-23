
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------- Project Specifications ----------------
FS = 48000               # Sampling frequency: 48,000 samples per second
FILTER_ORDER = 20        # Required filter order
RP = 1                   # Maximum passband ripple: 1 dB
RS = 17                  # Minimum stopband attenuation: 17 dB
FP = 22000               # Passband edge frequency: 22,000 Hz

# ------------- DESIGN THE ELLIPTIC IIR LOW-PASS FILTER ---------
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

    return sos     # Return the completed digital filter


# ----------- VERIFY AND PLOT THE FILTER FREQUENCY RESPONSE ---------------
def plot_frequency_response(sos):

    # Calculate the filter frequency response using a dense
    frequencies, response = signal.sosfreqz(
        sos, worN=2**22, fs=FS
    )
    # Convert the response magnitude into decibels.
    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-14)
    )

    # Find the first frequency after FP that reaches -17 dB
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
    
    # Print the numerical filter verification results.
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

# ---------------- CALCULATE THE FULL FREQUENCY RESPONSE FOR PLOTTING ----------------
    # Calculate a smaller frequency grid for the full response graph
    full_freq, full_response = signal.sosfreqz(
        sos, worN=16384, fs=FS
    )

    # Convert the response into decibels.
    full_db = 20 * np.log10(
        np.maximum(np.abs(full_response), 1e-14)
    )

    # Create two plots:
    # The first shows the full frequency range.
    # The second zooms into the transition region.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9)
    )

    # Add the main title to the figure
    fig.suptitle(
        "Elliptic IIR Low-Pass Filter Frequency Response",
        fontsize=16,
        fontweight="bold"
    )

# ------------ PLOT THE FULL FREQUENCY RESPONSE --------------
    
    # Blue line: actual filter frequency response.
    ax1.plot(
        full_freq, full_db,
        color="blue", linewidth=1.5,
        label="Filter response"
    )

    # Green horizontal line: -1 dB passband limit.
    ax1.axhline(
        -RP, color="green", linestyle="--",
        label="Passband limit (-1 dB)"
    )

    # Red horizontal line: -17 dB stopband limit.
    ax1.axhline(
        -RS, color="red", linestyle="--",
        label="Stopband limit (-17 dB)"
    )

    # Vertical line marking the passband edge.
    ax1.axvline(
        FP, color="green", linestyle=":",
        label="Passband edge (22,000 Hz)"
    )

    # Vertical line marking the calculated stopband edge.
    ax1.axvline(
        stopband_edge, color="red", linestyle=":",
        label=f"Stopband edge ({stopband_edge:.3f} Hz)"
    )

    # Light green shading: passband region.
    ax1.axvspan(
        0, FP, color="green", alpha=0.07
    )

    # Orange shading: transition region.
    ax1.axvspan(
        FP, stopband_edge, color="orange", alpha=0.45
    )

    # Light red shading: stopband region.
    ax1.axvspan(
        stopband_edge, FS / 2,
        color="red", alpha=0.05
    )

    # Add graph labels, limits, grid, and legend
    ax1.set_title("Full Frequency Response")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlim(0, FS / 2)
    ax1.set_ylim(-30, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

# ---------------- ZOOM INTO THE TRANSITION REGION ----------------

    # Define the frequency range around the transition.
    zoom_start = FP - 0.2
    zoom_end = stopband_edge + 0.2

    # Select only the frequency values inside the zoomed range
    zoom_mask = (
        (frequencies >= zoom_start) &
        (frequencies <= zoom_end)
    )

    # Plot the filter response in the zoomed region.
    ax2.plot(
        frequencies[zoom_mask],
        magnitude_db[zoom_mask],
        color="blue", linewidth=1.5,
        label="Filter response"
    )

    # Display the passband and stopband limits.
    ax2.axhline(
        -RP, color="green", linestyle="--",
        label="-1 dB passband limit"
    )

    ax2.axhline(
        -RS, color="red", linestyle="--",
        label="-17 dB stopband limit"
    )

    # Mark the passband and stopband edges.
    ax2.axvline(
        FP, color="green", linestyle=":",
        label="Passband edge"
    )

    ax2.axvline(
        stopband_edge, color="red", linestyle=":",
        label="Stopband edge"
    )

    # Highlight the transition region.
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

# ----------  RUN THE FILTER DESIGN PROGRAM -------------------

# This section runs only when this file is executed directly.
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