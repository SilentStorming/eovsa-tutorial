import os
import numpy as np
from matplotlib import gridspec as gridspec
from sunpy import map as smap
from astropy.time import Time
from matplotlib import pyplot as plt
from taskinit import qa, tb
from ptclean3_cli import ptclean3_cli as ptclean
from suncasa.utils import mstools
import multiprocessing
import astropy.units as u
from suncasa.utils import plot_mapX as pmX
import matplotlib.colors as colors
import matplotlib.dates as mdates
from tqdm import tqdm
from suncasa.utils import DButil
from scipy.signal import medfilt
from astropy.io import fits


def sfu2tb(freq, flux, beamsize):
    import scipy.constants as sc
    # frequency in Hz
    # flux in sfu
    # beamsize: FWHM of the circular beams in arcsec
    sfu2cgs = 1e-19
    vc = sc.c * 1e2
    kb = sc.k * 1e7
    sr = np.pi * (np.array(beamsize) / 206265.e0 / 2.e0) ** 2

    return np.array(flux) * sfu2cgs * vc ** 2e0 / (2e0 * kb * np.array(freq) ** 2e0 * sr)


'''
Example script for doing a time series of 30-band spectral imaging in a given time interval

History:
    2019-May-17 SY
        Created a new example script based on S. Yu's practice for imaging 
        the 2017 Aug 21 20:20 UT flare data. Made it available for EOVSA tutorial at 
        RHESSI XVIII Workshop (http://rhessi18.umn.edu/)
'''

################### USER INPUT GOES IN THIS BLOK ########################
vis = 'IDB20170821201800-202300.4s.slfcaled.ms'  # input visibility data
specfile = vis + '.dspec.npz'  ## input dynamic spectrum
nthreads = max(multiprocessing.cpu_count() - 2, 1)  # Number of processing threads to use
overwrite = True  # whether to overwrite the existed fits files.
trange = ''  # select the time range for imaging, leave it blank for using the entire time interval in the data
twidth = 2  # make one image out of every 2 time integrations
xycen = [380., 50.]  # define the center of the output map, in solar X and Y. Unit: arcsec
xran = [280., 480.]  # plot range in solar X. Unit: arcsec
yran = [-50., 150.]  # plot range in solar Y. Unit: arcsec
antennas = '!13;!14;!15'  # use all 13 EOVSA antennas. If some antenna is no good, drop it in this selection
npix = 128  # number of pixels in the image
cell = '2arcsec'  # pixel scale in arcsec
pol = 'XX'  # polarization to image, use XX for now
pbcor = True  # correct for primary beam response?
grid_spacing = 5. * u.deg  # spacing for plotting longitude and latitude grid in degrees
outdir = './images_series/'  # Specify where you want to save the output fits files
imresfile = 'imres.npz'  # File to write the imageing results summary
if not os.path.exists(outdir):
    os.makedirs(outdir)
outimgpre = 'EO'  # Something to add to the image name
#################################################################################


################### CONVERT XYCEN TO PHASE CENTER IN RA AND DEC ##################
try:
    phasecenter, midt = mstools.calc_phasecenter_from_solxy(vis, timerange=trange, xycen=xycen)
    print('use phasecenter: ' + phasecenter)
except:
    print('Provided time format not recognized by astropy.time.Time')
    print('Please use format trange="yyyy/mm/dd/hh:mm:ss~yyyy/mm/dd/hh:mm:ss"')
#################################################################################


################### (OPTIONAL) DEFINE RESTORING BEAM SIZE  ##################
### (Optional) the following is to define beam size ###
# read frequency information from visibility data
tb.open(vis + '/SPECTRAL_WINDOW')
reffreqs = tb.getcol('REF_FREQUENCY')
bdwds = tb.getcol('TOTAL_BANDWIDTH')
cfreqs = reffreqs + bdwds / 2.
tb.close()
spws = [str(s + 1) for s in range(30)]
sbeam = 30.
bmsz = [max(sbeam * cfreqs[1] / f, 6.) for f in cfreqs]
#################################################################################


################### MAIN BLOCK FOR CLEAN  ##################
midtstr = ((midt.isot).replace(':', '')).replace('-', '')
imname0 = outdir + '/' + outimgpre
fitsfiles = []
clnres = []
for s, sp in enumerate(spws):
    print('cleaning spw {0:s} with beam size {1:.1f}"'.format(sp, bmsz[s]))
    cfreq = cfreqs[int(sp)]
    res = ptclean(vis=vis,
                  imageprefix=outdir + '/' + outimgpre,
                  imagesuffix="_S{}".format(sp.zfill(2)),
                  antenna=antennas,
                  ncpu=nthreads,
                  twidth=twidth,
                  doreg=True,
                  usephacenter=True,
                  toTb=True,
                  overwrite=overwrite,
                  spw=sp,
                  timerange=trange,
                  specmode="mfs",
                  niter=500,
                  gain=0.05,
                  deconvolver="hogbom",
                  interactive=False,
                  mask='',
                  imsize=[npix],
                  cell=cell,
                  pbcor=pbcor,
                  datacolumn='data',
                  phasecenter=phasecenter,
                  stokes=pol,
                  weighting="briggs",
                  robust=0.0,
                  restoringbeam=[str(bmsz[s]) + 'arcsec'])
    clnres.append(res)

for s, sp in enumerate(spws):
    clnres[s]['Freq'] = [cfreqs[s] / 1.e9] * len(clnres[s]['ImageName'])
np.savez(outdir + imresfile, imres=clnres)
# imageing results summary are saved to "imresfile"
#################################################################################


########### PLOT ALL THE FITS IMAGES USING SUNPY.MAP ###########
imres = np.load(outdir + imresfile, encoding="latin1")
imres = imres['imres']
imsize = [npix] * 2
spws = [str(s + 1) for s in range(30)]

gs1 = gridspec.GridSpec(5, 6)
gs1.update(left=0.00, right=1.0, wspace=0.00, hspace=0.00, bottom=0.2, top=1.0)
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left=0.07, right=0.98, wspace=0.00, hspace=0.00, bottom=0.05, top=0.18)
fig = plt.figure(figsize=(10, 10.4))
im = []
axs = []
for s in range(len(spws)):
    axs.append(fig.add_subplot(gs1[s]))
ax_dspec = fig.add_subplot(gs2[0])

## plot the dynamic spectrum
ax = ax_dspec
specdata = np.load(specfile)
spec = specdata['spec']
(npol, nbl, nfreq, ntim) = spec.shape
spec = spec[0, 0, :, :]
tim = specdata['tim']
freq = specdata['freq']
freqghz = freq / 1e9
spec_tim = Time(tim / 24 / 3600, format='mjd')
spec_tim_plt = spec_tim.plot_date
dt = np.nanmean(np.diff(spec_tim_plt))
spec_tim_plt = np.hstack([spec_tim_plt - dt / 2.0, spec_tim_plt[-1] + dt / 2.0])
df = np.nanmean(np.diff(freqghz))
freqghz_plt = np.hstack([freqghz - df / 2.0, freqghz[-1] + df / 2.0])
spec_plt = spec[:, :]
vmax, vmin = spec_plt.max() * 1.0, spec_plt.max() * 0.01
norm = colors.LogNorm(vmax=vmax, vmin=vmin)
ax.pcolormesh(spec_tim_plt, freqghz_plt, spec_plt, norm=norm)
ax.set_ylabel('Frequency [GHz]')
ax.set_xlim(spec_tim_plt[0], spec_tim_plt[-1])
ax.set_ylim(freqghz_plt[0], freqghz_plt[-1])
date_format = mdates.DateFormatter('%H:%M:%S')
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
tidx = 0
tim_axvspan = ax.axvspan(Time(imres[0]['BeginTime'][tidx]).plot_date, Time(imres[0]['EndTime'][tidx]).plot_date,
                         facecolor='w', alpha=0.6)
ax.set_xlabel('Start Time ({})'.format(spec_tim[tidx].datetime.strftime('%b-%d-%Y %H:%M:%S')))
## get the spectrum at the peak time
tidxmax = np.nanargmax(np.nanmean(spec_plt, axis=0))
spec_v = medfilt(spec_plt[:, tidxmax], len(spec[:, 0]) / 100 * 2 + 1)
## get the dmaxs for images plot
spec_v_ = np.interp(cfreqs / 1e9, freqghz, spec_v) / 1e4  ## flux density in sfu
dmaxs = sfu2tb(cfreqs, spec_v_, bmsz)

for tidx in tqdm(range(len(imres[0]['ImageName']))):
    for s, sp in enumerate(spws):
        cfreq = imres[s]['Freq'][0]
        ax = axs[s]
        fsfile = imres[s]['ImageName'][tidx]
        if os.path.exists(fsfile):
            hdu = fits.open(fsfile)
            data = hdu[0].data.reshape(imsize)
            data[np.isnan(data)] = 0.0
            eomap = smap.Map(data, hdu[0].header)
            hdu.close()
            if tidx == 0:
                eomap_ = pmX.Sunmap(eomap)
                im.append(eomap_.imshow(axes=ax, vmax=dmaxs[s], vmin=dmaxs[s]*0.05))
                eomap_.draw_limb(axes=ax)
                eomap_.draw_grid(axes=ax, grid_spacing=grid_spacing)
                ax.set_title(' ')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_xlim(xran)
                ax.set_ylim(yran)
            else:
                im[s].set_array(eomap.data)
            if tidx == 0:
                ax.text(0.95, 0.95, '{0:.1f} GHz'.format(cfreq), transform=ax.transAxes, ha='right', va='top',
                        color='w', fontweight='bold')
        else:
            im[s].set_array(eomap.data * 0)
    if tidx == 0:
        ax = axs[0]
        timetext = ax.text(0.95, 0.02, eomap.date.strftime('%H:%M:%S'), ha='right', va='bottom',
                           transform=ax.transAxes, color='w', fontweight='bold')
    else:
        timetext.set_text(eomap.date.strftime('%H:%M:%S'))

    tim_axvspan_xy = tim_axvspan.get_xy()
    tim_axvspan_xy[np.array([0, 1, 4]), 0] = Time(imres[0]['BeginTime'][tidx]).plot_date
    tim_axvspan_xy[np.array([2, 3]), 0] = Time(imres[0]['EndTime'][tidx]).plot_date
    tim_axvspan.set_xy(tim_axvspan_xy)
    # tim_axvspan.set_xy(tim_axvspan_xy)
    figname = fsfile[:-9] + '.png'
    fig.savefig(figname,dpi=150)

plt.ion()

DButil.img2html_movie('{}/EO'.format(outdir))
