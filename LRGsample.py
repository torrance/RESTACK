from __future__ import print_function, division

from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord, Distance
import astropy.units as units
from astropy.units import Quantity
import matplotlib.pyplot as plt
import numpy as np


try:
    lrgs = np.load("LRGs-Lopes2007.npy")
except IOError:
    lrgs = np.loadtxt("photoz_grgiri_lrgc_allgals_10000random_weightn.dat")
    np.save("LRGs-Lopes2007.npy", lrgs)

print("Initial LRG count:", len(lrgs))
print(lrgs[0])

lrgs = lrgs[lrgs[:, 12] > 0]
print("After excluding negative redshift values:", len(lrgs))

ras, decs, zs = lrgs.T[[0, 1, 12]]

# Filter out the very near
idx = zs > 0.01
ras, decs, zs = ras[idx], decs[idx], zs[idx]

maxdist = Quantity(7.5, "Mpc")
mindist = Quantity(1, "Mpc")
minangle = Quantity(20, "arcminute")
maxangle = Quantity(180, "arcminute")

# When constructing coordinates, we make sure to use comoving metric as per V2021.
coords = SkyCoord(ras, decs, distance=Planck15.comoving_distance(zs), unit=("degree", "degree"))

# search_around_3d converts coords to cartesian points using the distance provided as the radial distance.
# This implcitly assumes a flat Universe. We hope the Universe is flat.
idx1, idx2, d2d, d3d = coords.search_around_3d(coords, maxdist)
print("Initial pair count:", len(idx1))

# Exclude close proximity LRGs
idx = d3d <= mindist
idx1, idx2, d2d, d3d = idx1[~idx], idx2[~idx], d2d[~idx], d3d[~idx]
print("Pairs remaining after mindist condition:", len(idx1))

# Exclude small angular separations
idx = d2d <= minangle
idx1, idx2, d2d, d3d = idx1[~idx], idx2[~idx], d2d[~idx], d3d[~idx]
print("Pairs remaining after minangle condition:", len(idx1))

# Exclude large angular separations
idx = d2d > maxangle
idx1, idx2, d2d, d3d = idx1[~idx], idx2[~idx], d2d[~idx], d3d[~idx]
print("Pairs remaining after maxangle condition:", len(idx1))

# Each LRG pair will exist twice, p1 <-> p2 and p2 <--> p1.
# Remove duplicates by excluding all pairs with exactly the same separation.
print("Before dedup:", len(idx1))
_, idx = np.unique(d3d, return_index=True)
idx1, idx2, d2d, d3d = idx1[idx], idx2[idx], d2d[idx], d3d[idx]
print("After dedup:", len(idx1))

lrgpairs = np.array([ras[idx1], decs[idx1], zs[idx1], ras[idx2], decs[idx2], zs[idx2]]).T
np.save("LRG-pairs-maxdist%g.npy" % maxdist.to_value("Mpc"), lrgpairs)

# # Output halo pairs as DS9 region file
# with open("LRG-pairs.reg", "w") as f:
#     print('global color=green font="helvetica 10 normal roman" edit=1 move=1 delete=1 highlite=1 include=1 wcs=wcs', file=f)
#     print("icrs", file=f)
#     for id1, id2 in zip(idx1, idx2):
#         if 165 < coords[id1].ra.deg < 195 and -12 < coords[id1].dec.deg < 18:
#             print("line %f %f %f %f # line=1 1" % (coords[id1].ra.deg, coords[id1].dec.deg, coords[id2].ra.deg, coords[id2].dec.deg), file=f)

plt.subplot(1, 3, 1)
midzs = 0.5 * (zs[idx1] + zs[idx2])
plt.hist(midzs, np.arange(0, 1.01, 0.025))

plt.subplot(1, 3, 2)
plt.hist(d2d.to_value("arcminute"), np.arange(20, 181, 10))

plt.subplot(1, 3, 3)
plt.hist(d3d.to_value("Mpc"), np.arange(1, 15.1, 1))
plt.show()

ras = coords.ra.wrap_at("180d").deg

plt.subplot(1, 1, 1, projection="aitoff")
plt.grid(True)
plt.scatter(np.radians(ras[idx1]), np.radians(decs[idx1]), s=1)
plt.show()
