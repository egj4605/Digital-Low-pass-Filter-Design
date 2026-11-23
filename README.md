**Overview: **

The purpose of this project is to design and implement a digital infinite Impulse Response (IIR) low-pass filter using the Elliptic filter design algorithm and the Bilinear transformation method.
A low-pass filter allows lower-frequency components of signals to pass while attenuating higher-frequency components. This is useful for reducing unwanted high-frequency noise in audio recordings.
For this project, an audio recording of SpongeBob's voice was used to demonstrate the filtering process. The audio is processed using the designed elliptic filter, and the original and filtered signals are compared in both the time domain and frequency domain.  

**Design Approach:**

To meet the project objectives, we designed a digital elliptic impulse-response low-pass filter using the bilinear transformation method. We used Python and the SciPy library to implement and test the audio.  
We set the filter parameters based on the project requirements: a 20th-order filter that operates at a 48000 Hz sampling rate, with a 22000 Hz passband edge, 1dB of passband ripple, and since the stopband attenuation was not specified, we chose 17dB as our assumption.
Next, we created an analog version of the filter. We applied frequency prewarping to prevent distortion, then we used the bilinear transformation to convert it into a digital filter. To ensure we are keeping our calculations stable, we broke down the filter into second-order sections.
To test the filter, we loaded a 48,000 Hz SpongeBob WAV file and random high-frequency noise. 
Finally, we ran the noise audio through our elliptic filter using standard forward filtering. By comparing the noisy and filtered audio with waveform plots, two-sided DFR graphs, and the filter’s frequency response, we then successfully confirmed that the filter removed the unwanted high-frequency noise. 

