import conversions as cnv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

plt.rc('font', size=35)
plt.rc('lines', lw=4)

bands = {
'g': 0,
'r': 1,
'i': 2,
'z': 3,
'y': 4,
'J': 5,
'H': 6,
'K': 7,
'S': 8}
band = str(sys.argv[1])
angle = int(sys.argv[2])

times_lums, times_mags, times_spec, lums, mags, spec, wav_lower, wav_upper = cnv.load_data()

# Test #1: Going from luminosities to magnitudes

converted_mags = cnv.lums_to_mags(lums[1:, :, :])

plt.figure(figsize=(19.2, 10.8))
plt.plot(times_mags, mags[bands[band], :, angle], c='k', label='sim data')
plt.plot(times_mags, converted_mags[bands[band], :, angle], ls='--', c='r', label='converted')
plt.xlabel('Time (days)')
plt.ylabel('Mags')
plt.xscale('log')
plt.gca().invert_yaxis()
plt.gca().set_xticks([0.125, 0.5, 1, 2, 4, 8, 16])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
plt.legend()
plt.savefig('lums_to_mags_conversion_%s.png' % band)

del converted_mags
plt.close()

# Test #2: Going from magnitudes to luminosities

converted_lums = cnv.mags_to_lums(mags)

plt.figure(figsize=(19.2, 10.8))
plt.plot(times_mags, lums[bands[band]+1, :, angle], c='k', label='sim data')
plt.plot(times_mags, converted_lums[bands[band], :, angle], ls='--', c='r', label='converted')
plt.xlabel('Time (days)')
plt.ylabel('Lums')
plt.xscale('log')
plt.gca().set_xticks([0.125, 0.5, 1, 2, 4, 8, 16])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
plt.legend()
plt.savefig('mags_to_lums_conversion_%s.png' % band)

del converted_lums
plt.close()

# Test #3: Going from spectra to magnitudes

for time_idx in range(spec.shape[0]):
	data = cnv.spec_to_mags(wav_lower, spec[time_idx, :, angle+2], band) # this angle index needs to be +2 of the magnitude angle index
	time_idx_mags = data[0]
	time_idx_flux = data[1]
	try:
		converted_mags = np.append(converted_mags, time_idx_mags)
	except NameError:
		converted_mags = time_idx_mags

plt.figure(figsize=(19.2, 10.8))
plt.plot(times_mags, mags[bands[band], :, angle], c='k', label='sim data')
plt.plot(times_spec, converted_mags, ls='--', c='r', label='converted')
plt.xlabel('Time (days)')
plt.ylabel('Mags')
plt.xscale('log')
plt.gca().invert_yaxis()
plt.gca().set_xticks([0.125, 0.5, 1, 2, 4, 8, 16])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
plt.legend()
plt.savefig('spec_to_mags_conversion_%s.png' % band)

del converted_mags
plt.close()
