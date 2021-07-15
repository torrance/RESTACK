#! /usr/bin/env python
from __future__ import print_function, division

import argparse
from multiprocessing import Pool, Lock
from multiprocessing.shared_memory import SharedMemory
import os
import os.path
import re
from threading import Semaphore

from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
import astropy.units as units
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import numpy as np
from scipy.interpolate import interp2d, griddata, RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import rotate, zoom
from scipy.signal import fftconvolve

from astrobits.coordinates import fits_to_radec


def getexclusions(f):
    exclusions = []
    regex = re.compile("^circle\((.*)\)")
    for line in f:
        m = regex.search(line)
        if m is None:
            continue
        else:
            ra, dec, radius = m.group(1).split(",")
            coord = SkyCoord(ra, dec, unit=("hourangle", "deg"))
            radius = Angle(radius)
            exclusions.append((coord, radius))

    return exclusions


def rotateandscale(origin, destshape, x0, y0, scale, theta):
    # Create origin coordinates
    # These are just 1D arrays of the grid axes as accepted by RegularGridInterpolator
    xs, ys = np.arange(origin.shape[0], dtype=float), np.arange(origin.shape[1], dtype=float)

    # Offset from origin
    xs -= x0
    ys -= y0

    # Scale
    xs *= scale
    ys *= scale

    # Create destination coordinates
    xsprime, ysprime = np.indices(destshape, dtype=float)

    # Offset from origin
    xsprime -= destshape[0] / 2
    ysprime -= destshape[0] / 2

    # Rotate destination coordinates into coordinate system of xs, ys in preparation for interpolation
    xsprime, ysprime = xsprime * np.cos(theta) - ysprime * np.sin(theta), xsprime * np.sin(theta) + ysprime * np.cos(theta)

    # And finally, interpolate
    f = RegularGridInterpolator((xs.reshape(-1), ys.reshape(-1)), origin, bounds_error=False)
    return f(np.array([xsprime, ysprime]).T, method="linear").reshape(*destshape)


# Run this at pool initialization to ensure writelock is available to all processes
def initpool(l):
    global writelock
    writelock = l


# This makes the callback upon worker completion to release the semaphore.
# We wrap this in a closure to pass in additional state to the callback.
def workercallback(_):
    semaphore.release()


def makestack(x1, y1, x2, y2, imgshr, imgshape, stackshr, stackshape):
    img = np.ndarray(imgshape, np.float64, buffer=imgshr.buf)
    stack = np.ndarray(stackshape, np.float64, buffer=stackshr.buf)

    # Distance
    rpx = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    print("Pixel distance:", rpx)

    # Scale factor
    scalefactor = abs(maxrpx / rpx)
    print("Scale factor:", scalefactor)

    # Calculate rotation angle
    theta = np.arctan2(y2 - y1, x2 - x1)
    print("Rotation angle:", theta)

    # Central pixel
    x0, y0 = x1 + 0.5 * rpx * np.cos(theta), y1 + 0.5 * rpx * np.sin(theta)
    print("Central pixel:", x0, y0)

    img = rotateandscale(img, stack.shape, x0, y0, scalefactor, theta)

    # All NaN (and inf) values set to zero in both image and weight maps.
    img[~np.isfinite(img)] = 0

    # Stack is shared amongst all processes, and we need to ensure serial writes
    with writelock:
        stack[:] += img

    imgshr.close()
    stackshr.close()

    return None


parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--lrgpairs", required=True)
parser.add_argument("--prefix", required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--noexclusions", action="store_true")
parser.add_argument("--cores", type=int, default=os.cpu_count())
parser.add_argument("--maxrpx", type=int, default=600)
args = parser.parse_args()

LRGs = np.load(args.lrgpairs)  # [ra1, dec1, z1, ra2, dec2, z2]

LRGindex = np.zeros(len(LRGs), dtype=bool)
print("Initial LRG pairs loaded:", len(LRGs))

coords1 = SkyCoord(LRGs[:, 0], LRGs[:, 1], unit="deg")
coords2 = SkyCoord(LRGs[:, 3], LRGs[:, 4], unit="deg")

# Convert to pixel coordinates of respective image
wcs = WCS(fits.getheader(args.image))
coords1_px = skycoord_to_pixel(coords1, wcs)
coords2_px = skycoord_to_pixel(coords2, wcs)

# Set maxrpx
rs_px = np.sqrt((coords1_px[0] - coords2_px[0])**2 + (coords1_px[1] - coords2_px[1])**2)
maxrpx = max(rs_px)
minrpx = min(rs_px)

print("Maximum distance:", maxrpx)
print("Minimum distance:", minrpx)
assert maxrpx < args.maxrpx
maxrpx = args.maxrpx

# Create array of halo pairs in pixel coordinates
LRGs_px = np.array([coords1_px[0], coords1_px[1], coords2_px[0], coords2_px[1]]).T

# Allow resuming stacking if we reach timelimits for our jobs.
if args.resume:
    stackedHDU = fits.open(args.prefix + "-stacked.fits")[0]
    stacked = np.empty(stackedHDU.data.shape)
    stacked[:] = stackedHDU.data.T
    stackedweights = np.zeros_like(stacked)
    stackedweights[:] = fits.getdata(args.prefix + "-weights.fits").T
    stacked *= stackedweights
    stacked[~np.isfinite(stacked)] = 0
    LRGindex = np.load(args.prefix + "-LRGindex.npy")

    kstart = stackedHDU.header["KNEXT"]
    N = stackedHDU.header["NITER"]
else:
    # Create the stacked image template
    stacked = np.zeros((int(4 * maxrpx), int(4* maxrpx)))
    stackedweights = np.zeros_like(stacked)

    stackedHDU = fits.PrimaryHDU(data=stacked.copy())
    stackedHDU.header["PEAK1"] = 4 * maxrpx / 2 - maxrpx / 2
    stackedHDU.header["PEAK2"] = 4 * maxrpx / 2 + maxrpx / 2

    kstart = 0
    N = 0

if kstart >= len(LRGs_px):
    print("Already finished!")
    exit()

stackedx0, stackedy0 = stacked.shape[0] // 2, stacked.shape[1] // 2
print("Stacked shape:", stacked.shape, "stackedx0:", stackedx0, "stackedy0:", stackedy0)

# Load residual and weight images
prefix = args.image[:-len("-restored-fluxxed.fits")]
_residual = np.squeeze(fits.open(prefix + "-residuals-fluxxed.fits")[0].data).T
_beamweight = np.sqrt(np.squeeze(fits.open(prefix + "-weight-fluxxed.fits")[0].data).T)

# Turn fits arrays into real numpy arrays
residual = np.empty(_residual.shape)
residual[:] = _residual
beamweight = np.empty(_beamweight.shape)
beamweight[:] = _beamweight

if not args.noexclusions:
    # Apply exclusions
    ras, decs = fits_to_radec(fits.open(args.image)[0])
    coords = SkyCoord(ras.flatten(), decs.flatten(), unit=("radian", "radian"))
    with open(os.path.dirname(args.image) + "exclusions.reg") as f:
        exclusions = getexclusions(f)
    excludedidx = np.zeros(ras.size, dtype=bool)
    for excoord, radius in exclusions:
        excludedidx = np.any([excludedidx, excoord.separation(coords) <= radius], axis=0)
    excludedidx = np.reshape(excludedidx, residual.shape)
    print(excludedidx.shape)

    residual[excludedidx.T] = np.nan
    beamweight[excludedidx.T] = 0

# Weight beam response by average image noise
constantnoise = residual * beamweight
imgrms = 1.4826 * np.nanmedian(abs(constantnoise - np.nanmedian(constantnoise)))
noisemap = imgrms / beamweight
print("Image rms:", np.nanmin(noisemap), np.nanmax(noisemap))

# Set weight as square of itself
weight = 1 / noisemap**2
weight[~np.isfinite(weight)] = 0

print("Max weighting:", weight.max())

# Apply weights to image
residual *= weight

# Stack using a worker pool
# We also create a semaphore to ensure that we don't enqueue the pool beyond our memory allowance
writelock = Lock()
semaphore = Semaphore(args.cores + 2)
pool = Pool(args.cores, initializer=initpool, initargs=(writelock,))
asyncresults = []

# Create all maps as shared memory
residualshr = SharedMemory(create=True, size=residual.nbytes)
residualshrnd = np.ndarray(residual.shape, dtype=residual.dtype, buffer=residualshr.buf)
residualshrnd[:] = residual
weightshr = SharedMemory(create=True, size=weight.nbytes)
weightshrnd = np.ndarray(weight.shape, dtype=weight.dtype, buffer=weightshr.buf)
weightshrnd[:] = weight
stackedshr = SharedMemory(create=True, size=stacked.nbytes)
stackedshrnd = np.ndarray(stacked.shape, dtype=stacked.dtype, buffer=stackedshr.buf)
stackedshrnd[:] = stacked
stackedweightsshr = SharedMemory(create=True, size=stackedweights.nbytes)
stackedweightsshrnd = np.ndarray(stackedweights.shape, dtype=stackedweights.dtype, buffer=stackedweightsshr.buf)
stackedweightsshrnd[:] = stackedweights

for k, (x1, y1, x2, y2) in enumerate(LRGs_px[kstart:], kstart):
    print("\nProgress: %.1f%%" % (100 * k / len(LRGs_px)), "N:", N)
    print("Coords:", (x1, y1), (x2, y2))

    x1px, y1px, x2px, y2px = np.rint([x1, y1, x2, y2]).astype(int)
    # We explicitly test for negative pixel indices, since in Python these are valid.
    if np.any(np.isnan([x1px, y1px, x2px, y2px])) or x1px < 0 or y1px < 0 or x2px < 0 or y2px < 0:
        print("Skipping: outside image")
        continue
    try:
        if np.isnan(residual[y1px, x1px]) or np.isnan(residual[y2px, x2px]):
            print("Skipping: LRG peaks are NaN")
            continue
    except IndexError:
        print("Skipping: outside image")
        continue

    N += 1
    LRGindex[k] = True
    for imgshr, stackshr in [(residualshr, stackedshr), (weightshr, stackedweightsshr)]:
        semaphore.acquire()
        asyncresults.append(pool.apply_async(
            makestack,
            (x1, y1, x2, y2, imgshr, residual.shape, stackshr, stacked.shape),
            callback=workercallback,
        ))
        # makestack(x1, y1, x2, y2, imgshr, residual.shape, stackshr, stacked.shape)

    if (N & (N - 1)) == 0 and N > 100:
        # Wait for jobs to complete
        [res.wait() for res in asyncresults]

        stackedHDU.header["NITER"] = N
        stackedHDU.data = (stackedshrnd / stackedweightsshrnd).T
        stackedHDU.writeto(args.prefix + "-niter%d-stacked.fits" % N, overwrite=True)
        stackedHDU.data = stackedweightsshrnd.T
        stackedHDU.writeto(args.prefix + "-niter%d-weights.fits" % N, overwrite=True)

    if N % 1000 == 0:
        # Wait for jobs to complete
        [res.wait() for res in asyncresults]

        stackedHDU.header["NITER"] = N
        stackedHDU.header["KNEXT"] = k + 1
        stackedHDU.data = (stackedshrnd  / stackedweightsshrnd).T
        stackedHDU.writeto(args.prefix + "-stacked.fits", overwrite=True)
        stackedHDU.data = stackedweightsshrnd.T
        stackedHDU.writeto(args.prefix + "-weights.fits", overwrite=True)
        np.save(args.prefix + "-LRGindex.npy", LRGindex)

# Wait for all workers to finish running
pool.close()
pool.join()

stackedHDU.header["NITER"] = N
stackedHDU.header["KNEXT"] = k + 1
stackedHDU.data = (stackedshrnd / stackedweightsshrnd).T
stackedHDU.writeto(args.prefix + "-stacked.fits", overwrite=True)
stackedHDU.data = (stackedweightsshrnd).T
stackedHDU.writeto(args.prefix + "-weights.fits", overwrite=True)
np.save(args.prefix + "-LRGindex.npy", LRGindex)

residualshr.close()
residualshr.unlink()
weightshr.close()
weightshr.unlink()
stackedshr.close()
stackedshr.unlink()
stackedweightsshr.close()
stackedweightsshr.unlink()
