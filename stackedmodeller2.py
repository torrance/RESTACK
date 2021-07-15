from __future__ import print_function, division

import argparse
import os.path

from astropy.io import fits
import matplotlib
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.sans-serif"] = "Times New Roman"
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from astrobits.correlation import radialautocorrelation


@njit()
def radialaverage(data, rs, bins):
    data = data.flatten()
    rs = rs.flatten()

    idxs = np.digitize(rs, bins)

    mus = np.zeros(len(bins) - 1)
    mus2 = np.zeros(len(bins) - 1)
    Ns = np.zeros(len(bins) - 1)
    for i, d in zip(idxs, data):
        if not np.isfinite(d) or i == 0 or i == len(bins):
            continue

        mus[i - 1] += d
        mus2[i - 1] += d**2
        Ns[i - 1] += 1

    mus /= Ns
    mus2 /= Ns

    return mus, np.sqrt(mus2 - mus**2), Ns


parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

prefix = os.path.splitext(os.path.basename(args.image))[0]
print("Prefix:", prefix)

hdu = fits.open(args.image)[0]
window = np.empty(hdu.data.shape)
window[:] = hdu.data.T * 1E6 # Convert to [x, y] ordering, and native numpy array
xcenter, ycenter = window.shape[1] // 2, window.shape[0] // 2
scalepx = (hdu.header["PEAK2"] - hdu.header["PEAK1"]) / 2

print(repr(hdu.header))

# NaN high
ys, xs = np.meshgrid(range(window.shape[0]), range(window.shape[1]))
rs = np.sqrt((xs - xcenter)**2 + (ys - ycenter)**2)
idx = rs > 4 * scalepx
window[idx] = np.nan

# Subtract out background
idx = rs > 2 * scalepx
window -= np.nanmean(window[idx])

# Model peaks
bins = np.arange(-0.5, xcenter, 1)
mids = 0.5 * (bins[:-1] + bins[1:])
modelsum = np.zeros_like(window)
peakregions = np.zeros(window.shape, dtype=bool)

for npeak, ypeak in enumerate([int(hdu.header["PEAK1"]), int(hdu.header["PEAK2"])]):
    print("Modelling peak:", xcenter, ypeak, "Value:", window[xcenter, ypeak])
    print("Actual peak value:", window[xcenter - 100:xcenter + 100, ypeak - 100:ypeak + 100].max())

    rs = np.sqrt((xs - xcenter)**2 + (ys - ypeak)**2)
    thetas = np.arctan2(ys - ypeak, xs - xcenter)

    peakregions[rs < 0.2 * scalepx] = True

    if npeak == 0:
        idx = np.any([thetas <= 0, thetas == np.pi], axis=0)
    elif npeak == 1:
        idx = np.any([thetas >= 0, thetas == -np.pi], axis=0)
    # if npeak == 0:
    #     idx = np.all([thetas >= np.radians(-90 - 45), thetas <= np.radians(-90 + 45)], axis=0)
    #     idx = np.any([idx, thetas == 0], axis=0)
    # elif npeak == 1:
    #     idx = np.all([thetas >= np.radians(90 - 45), thetas <= np.radians(90 + 45)], axis=0)
    #     idx = np.any([idx, thetas == 0], axis=0)

    exterior = window[idx]
    exteriorrs = rs[idx]

    # window[idx] = 0
    # plt.imshow(window, origin="bottom")
    # plt.show()

    mus, _, Ns = radialaverage(exterior, exteriorrs, bins)
    print(mus[:5])

    # Blank out mus where Ns suddenly begins to decrease
    idx = np.argmax(Ns)
    mus[idx:] = np.nan

    plt.plot(mids, mus)
    # plt.show()

    # Interpolating model...
    model = interp1d(mids[np.isfinite(mus)], mus[np.isfinite(mus)], bounds_error=False, fill_value=np.nan)(rs)
    model[np.isnan(model)] = 0

    modelsum += model

delta = window - modelsum
deltablanked = delta.copy()
deltablanked[peakregions] = np.nan
stderr = np.nanstd(delta[xcenter - int(2 * scalepx):xcenter + int(2 * scalepx), ycenter - int(2 * scalepx):ycenter + int(2 * scalepx)])
print("Estimated standard error:", stderr)

with open("output/" + prefix + "-noise.txt", "w") as f:
    print(stderr, file=f)

np.save("output/" + prefix + "-1Dwindow.npy", window[xcenter, :])
np.save("output/" + prefix + "-1Dmodelsum.npy", modelsum[xcenter, :])

# # Plot noise characteristics
# plt.figure(figsize=(5, 5))
# plt.subplot(2, 1, 1)
# bins = np.linspace(-5 * stderr, 5 * stderr, 100)
# mids = (bins[:-1] + bins[1:]) / 2

# ns, _, _ = plt.hist(deltablanked[np.isfinite(deltablanked)], bins=bins, density=True, color="dodgerblue", alpha=1)

# def gaussian(xs, A, x0, sigma):
#     return A * np.exp(-(xs - x0)**2 / (2 * sigma**2))

# popt, pcov = curve_fit(gaussian, mids, ns, p0=(max(ns), 0, stderr))
# print(popt)
# plt.plot(mids, gaussian(mids, *popt), color="black", linestyle="dashed")
# plt.xlim([min(mids), max(mids)])
# plt.xlabel("Pixel value [$\mu$Jy beam$^{-1}$]")
# plt.ylabel("Normalised counts")

# Calculate autocorrelation
deltablankedzeros = delta[xcenter - int(2 * scalepx):xcenter + int(2 * scalepx), ycenter - int(2 * scalepx):ycenter + int(2 * scalepx)].copy()
deltablankedzeros -= np.nanmean(deltablankedzeros)
sigma2 = np.nanstd(deltablankedzeros)**2
Ns = np.isfinite(deltablankedzeros).astype(float)
deltablankedzeros[~np.isfinite(deltablankedzeros)] = 0
bins = np.arange(-0.5, 400, 1)
mids = (bins[:-1] + bins[1:]) / 2
mus, _, _, _ = radialautocorrelation(deltablankedzeros, bins, sigma2=sigma2, Ns=Ns)

hwhm = interp1d(mus, mids)(0.5)
fwhm = (hwhm * 2) / scalepx  # Convert to normalised units
print("FWHM:", fwhm)

plt.subplot(2, 1, 2)
plt.plot(mids, mus)
xlabels = np.arange(0, 1.01, 0.05)
xvals = xlabels * scalepx
idx = xvals < 400
xvals, xlabels = xvals[idx], xlabels[idx]
plt.xticks(xvals, ["%g" % x for x in xlabels])
plt.vlines(hwhm, 0, 0.5, colors="black", linestyles="dotted")
plt.hlines(0.5, 0, hwhm, colors="black", linestyles="dotted")
plt.xlim([0, 400])
plt.ylim([0, 1])
plt.xlabel("Radial offset [normalised distance]")
plt.ylabel("Autocorrelation")

plt.tight_layout()
plt.savefig("output/" + prefix + "-noise.pdf")
plt.savefig("output/" + prefix + "-noise.png")
# plt.show()

plt.figure(figsize=(14, 3.5))
plt.subplot(1, 3, 1)
plt.imshow(window, vmin=np.nanpercentile(window, 0.01), vmax=np.nanpercentile(window, 99.99), origin="lower", cmap="plasma")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.yticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.grid()
plt.xlim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.ylim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.colorbar(label="[$\mu$Jy beam$^{-1}$]")
plt.subplot(1, 3, 2)
plt.imshow(window, vmin=np.nanpercentile(window, 0.01), vmax=np.nanpercentile(window, 98), origin="lower", cmap="plasma")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.yticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.grid()
plt.xlim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.ylim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.colorbar(label="[$\mu$Jy beam$^{-1}$]")
plt.subplot(1, 3, 3)
plt.imshow(delta, vmin=-5 * stderr, vmax=5 * stderr, origin="lower", cmap="plasma")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.yticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.grid()
plt.xlim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.ylim([ycenter - 2.5 * scalepx, ycenter + 2.5 * scalepx])
plt.colorbar(label="[$\mu$Jy beam$^{-1}$]")
plt.savefig("output/" + prefix + "-beforeafter.pdf")
plt.savefig("output/" + prefix + "-beforeafter.png")
# plt.show()

plt.figure(figsize=(10, 4.5))

plt.subplot(3, 1, 1)
plt.plot(window[xcenter, :], color="dodgerblue", label="Stacked")
plt.fill_between(range(len(window[xcenter, :])), window[xcenter, :] - 3 * stderr, window[xcenter, :] + 3 * stderr, color="dodgerblue", alpha=0.3)
plt.plot(modelsum[xcenter, :], color="red", label="Model")
# plt.plot(np.load("output/150MHz-z0.00-999.00-convolvedMWA-noise0-webmodel-5mJycleaned-residual-nlimit20000-stacked-1Dwindow.npy"), color="dodgerblue", linestyle="dashed", label="Cosmic Web")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.xlim([ycenter - 3 * scalepx, ycenter + 3 * scalepx])
plt.grid()
plt.legend()
ymin, ymax = plt.ylim()
plt.hlines(ymin + 0.9 * (ymax - ymin), ycenter - 2.9 * scalepx, ycenter - (2.9 - fwhm) * scalepx)
plt.vlines(ycenter - 2.9 * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.vlines(ycenter - (2.9  - fwhm) * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.ylabel("Stack value [$\mu$Jy beam$^{-1}$]")

plt.subplot(3, 1, 2)
plt.plot(delta[xcenter, :] / stderr, color="forestgreen")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.xlim([ycenter - 3 * scalepx, ycenter + 3 * scalepx])
plt.grid()
ymin, ymax = plt.ylim()
plt.hlines(ymin + 0.9 * (ymax - ymin), ycenter - 2.9 * scalepx, ycenter - (2.9 - fwhm) * scalepx)
plt.vlines(ycenter -2.9 * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.vlines(ycenter - (2.9  - fwhm) * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.ylabel("Residual $(\Delta / \sigma)$")
plt.xlabel("Normalised cluster pair distance")

plt.subplot(3, 1, 3)
noisesamples = []
width = 0.2
for xoffset in range(int(-2 * scalepx), int(-1 * scalepx)):
    noisesamples.append(
        np.mean(delta[int(xcenter + xoffset - width * scalepx):int(xcenter + xoffset + width * scalepx), ycenter - int(2 * scalepx):ycenter + int(2 * scalepx)], axis=0)
    )
for xoffset in range(int(1 * scalepx), int(2 * scalepx)):
    noisesamples.append(
        np.mean(delta[int(xcenter + xoffset - width * scalepx):int(xcenter + xoffset + width * scalepx), ycenter - int(2 * scalepx):ycenter + int(2 * scalepx)], axis=0)
    )
widenoise = np.std(noisesamples)
print("Widenoise:", widenoise)
plt.plot(np.mean(delta[int(xcenter - width * scalepx):int(xcenter + width * scalepx), :], axis=0) / widenoise, color="forestgreen")
plt.xticks([ycenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
plt.xlim([ycenter - 3 * scalepx, ycenter + 3 * scalepx])
plt.grid()
ymin, ymax = plt.ylim()
plt.hlines(ymin + 0.9 * (ymax - ymin), ycenter - 2.9 * scalepx, ycenter - (2.9 - fwhm) * scalepx)
plt.vlines(ycenter -2.9 * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.vlines(ycenter - (2.9  - fwhm) * scalepx, ymin + 0.87 * (ymax - ymin), ymin + 0.93 * (ymax - ymin))
plt.ylabel("Integrated\nResidual $(\Delta / \sigma)$")
plt.xlabel("Normalised cluster pair distance")

plt.tight_layout()
plt.savefig("output/" + prefix + "-modelled.pdf")
plt.savefig("output/" + prefix + "-modelled.png")
plt.show()


# plt.subplot(1, 2, 2)
# plt.plot(np.nanmean(deltablanked[:, ycenter - int(1 * scalepx):ycenter + int(1 * scalepx)], axis=1))
# plt.xticks([xcenter - i * scalepx for i in range(-3, 4)], range(-3, 4))
# plt.xlim([xcenter - 3 * scalepx, xcenter + 3 * scalepx])
# plt.grid()
# # plt.show()

# # Null the peaks
# fits.PrimaryHDU(data=deltablanked.T, header=hdu.header).writeto("delta.fits", overwrite=True)
