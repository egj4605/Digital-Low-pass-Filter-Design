# 🎧 Digital Low-Pass Filter Design
### Signal Processing Project

**Filter Type:** Elliptic Infinite Impulse Response (IIR) Low-Pass Filter  
**Design Method:** Bilinear Transformation

## 📘 1. Project Overview

The purpose of this project is to design and implement a digital Infinite Impulse Response (IIR) low-pass filter using the Elliptic filter design algorithm and the Bilinear Transformation method.

A low-pass filter allows lower-frequency components of a signal to pass through while reducing higher-frequency components. This is useful for removing unwanted high-frequency noise from audio recordings.

For this project, we used an audio recording of SpongeBob's voice to demonstrate the filtering process. The audio was processed using our designed elliptic filter, and the original and filtered signals were compared in both the time domain and frequency domain.

## ⚙️ 2. Filter Design Specifications

The filter was designed using the following parameters:

| Parameter | Value |
|---|---|
| Filter Type | Elliptic Low-Pass |
| Filter Structure | IIR |
| Filter Order | 20 |
| Sampling Frequency | 48,000 Hz |
| Passband Edge Frequency | 22,000 Hz |
| Passband Ripple | 1 dB |
| Stopband Attenuation | 17 dB (assumed) |
| Design Method | Bilinear Transformation |

The stopband attenuation was not specified in the project requirements, so we selected 17 dB as our design assumption.

## 🛠️ 3. Design Approach

The filter was designed and implemented using Python and the SciPy library.

The main steps included:

1. **Filter Specifications:** Selected the filter order, sampling frequency, passband edge, and passband ripple based on the project requirements.

2. **Analog Filter Design:** Created an analog version of the elliptic filter.

3. **Frequency Prewarping:** Applied frequency prewarping to account for the frequency distortion introduced by the bilinear transformation.

4. **Bilinear Transformation:** Converted the analog filter into a digital filter.

5. **Second-Order Sections:** Broke the filter into second-order sections to help maintain numerical stability during calculations.

6. **Audio Processing:** Loaded a 48,000 Hz SpongeBob WAV audio recording and added random high-frequency noise for testing.

7. **Filtering and Comparison:** Processed the noisy audio through the designed elliptic filter using standard forward filtering.

8. **Results Analysis:** Compared the original, noisy, and filtered audio using waveform plots, two-sided frequency-domain plots, and the filter's frequency response.

## 💻 4. Tools and Libraries

The project used the following tools and libraries:

- **Python:** Main programming language used for implementation.
- **NumPy:** Used for numerical calculations and signal data.
- **SciPy:** Used for filter design and digital signal processing.
- **SoundFile:** Used for reading and writing WAV audio files.
- **Matplotlib:** Used to generate waveform plots and frequency-response graphs.

## 📊 5. Results

The designed elliptic low-pass filter was tested using the noisy SpongeBob audio recording.

The results were analyzed by comparing the audio signals before and after filtering. The waveform plots and frequency-domain graphs helped demonstrate how the filter reduced unwanted high-frequency noise.

The testing allowed us to evaluate the filter's performance and confirm that it successfully reduced the added high-frequency noise.

## 👥 6. Project Team

- **Evan Jorge-Munoz**
- **Nejwa Abdulmalik**
- **Maximiliano Barajas**
- **Johnny Ramirez**

All team members contributed to the development, testing, debugging, documentation, and presentation of the project. We also worked together to solve problems and understand different parts of the implementation.

## 🎯 7. Conclusion

This project demonstrates how an elliptic IIR low-pass filter can be designed and implemented using the bilinear transformation method.

Through the use of analog filter design, frequency prewarping, and second-order sections, we were able to develop a digital filter and test it using an audio recording with added high-frequency noise.

The project provided practical experience with digital filter design, audio signal processing, and analyzing filter performance using both time-domain and frequency-domain representations.

