'''
Ariyana Bonab
Comp Astrophysics HW 4

In this code, I am going to demonstrate how to take a fourier transform and plot the power spectrum, using
TESS data. To do this, I first loaded in the fits file. From here, I looked at the plotted light curve,
and chose a continuous observation epoch with a lot of data, choosing a denser region on my plot.

In order to handle missing data, I performed a linear interpolation. After this, I was able to perform a Fourier
transform, and compute & plot the power spectrum.

Lastly, to see how few coefficients I needed to use to capture the behavior of the eclipsing binary to get the
inverse transform, I found the dominant frequencies and reconstructed the signal using top N Fourier coefficients.

Lastly, I plotted my original light curve vs the reconstructed one.

'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate

# Loading the FITS file
filename = 'tic0001630690.fits' 
hdul = fits.open(filename)
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

# Showing the light curve plotted
plt.figure(figsize=(12, 4)) #making it bigger since data is so close together
plt.plot(times, fluxes, 'k.', markersize=5, color='crimson')
plt.xlabel('Time (days)')
plt.xlim(2309, 2318) # adjusting x limit range
plt.ylabel('Flux')
plt.title('TIC 0001630690')
plt.show()

# Select a continuous observation epoch with lots of data
# I chose a dense region in my plot 
#mask = (times > times[0]) & (times < times[0] + 30)
mask = (times > 1571) & (times < 1595) # after analyzing i found that 1571 to 1595 days worked best!
times_subset = times[mask]
fluxes_subset = fluxes[mask]

print(f"Selected {len(times_subset)} observations")
print(f"Time range: {times_subset[0]:.2f} to {times_subset[-1]:.2f} days")

# Handle missing data with linear interpolation
time_diffs = np.diff(times_subset)
median_dt = np.median(time_diffs)
print(f"Median time step: {median_dt:.6f} days")

# I created an evenly spaced grid of time
time_grid = np.arange(times_subset[0], times_subset[-1], median_dt)
flux_interp = np.interp(time_grid, times_subset, fluxes_subset)
print(f"Interpolated to {len(time_grid)} evenly spaced points")

# Now to complete the Fourier Transform!
fft_result = np.fft.fft(flux_interp)
frequencies = np.fft.fftfreq(len(time_grid), d=median_dt)

# I computed the power spectrum, only positive frequencies
power = np.abs(fft_result)**2
positive_freq_mask = frequencies > 0
freq_positive = frequencies[positive_freq_mask]
power_positive = power[positive_freq_mask]

# Plotting the power spectrum
plt.plot(freq_positive, power_positive)
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.xlim(4, 17)
plt.show()
# Find dominant frequencies
n_peaks = 10
peak_indices = np.argsort(power_positive)[-n_peaks:][::-1]
dominant_freqs = freq_positive[peak_indices]
dominant_powers = power_positive[peak_indices]

print(f"\nTop {n_peaks} dominant frequencies:")
for i, (freq, pwr) in enumerate(zip(dominant_freqs, dominant_powers)):
    period = 1/freq if freq > 0 else np.inf
    print(f"{i+1}. Frequency: {freq:.4f} 1/day, Period: {period:.4f} days, Power: {pwr:.2e}")

# Reconstructing the signal using top N Fourier coefficients
n_coeffs = 10  # I played around with this number a bit
fft_filtered = np.zeros_like(fft_result)
sorted_indices = np.argsort(power)[::-1]
fft_filtered[sorted_indices[:n_coeffs]] = fft_result[sorted_indices[:n_coeffs]]
fft_filtered[-sorted_indices[:n_coeffs]] = fft_result[-sorted_indices[:n_coeffs]] # this is to include negative frequencies
reconstructed = np.fft.ifft(fft_filtered).real

# Ploting the original vs reconstructed
plt.figure(figsize=(12, 4))
plt.plot(time_grid, flux_interp, '.', markersize=1, label='Original Data', alpha=0.5)
plt.plot(time_grid, reconstructed, 'r-', linewidth=1, label=f'Reconstructed ({n_coeffs} coefficients)')
plt.xlabel('Time (days)')
plt.ylabel('Flux')
plt.title(f'Light Curve Reconstruction with {n_coeffs} Fourier Coefficients')
plt.legend()
plt.show()
