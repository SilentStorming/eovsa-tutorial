import os, sys
import numpy as np
from matplotlib import gridspec as gridspec
from sunpy import map as smap
from astropy.time import Time
from matplotlib import pyplot as plt
from taskinit import qa, tb
from ptclean3_cli import ptclean3_cli as ptclean
from suncasa.utils import helioimage2fits as hf
import multiprocessing
import astropy.units as u
from suncasa.utils import plot_mapX as pmX
import matplotlib.colors as colors
import matplotlib.dates as mdates
from tqdm import tqdm
from suncasa.utils import DButil
from scipy.signal import medfilt
from astropy.io import fits
import psutil

'''
Example script for doing a time series of 30-band spectral imaging in a given time interval

History:
    2019-May-17 SY
        Created a new example script based on S. Yu's practice for imaging 
        the 2017 Aug 21 20:20 UT flare data. Made it available for EOVSA tutorial at 
        RHESSI XVIII Workshop (http://rhessi18.umn.edu/)
    2019-May-19 SY, BC
        Made changes to check if the script is being run on Virgo
        Cleaned up the script
'''

################### USER INPUT GOES IN THIS BLOK ########################
vis = 'IDB20170821201800-202300.4s.slfcaled.ms'  # input visibility data
specfile = vis + '.dspec.npz'  ## input dynamic spectrum
nthreads = 1  # Number of processing threads to use
overwrite = True  # whether to overwrite the existed fits files.
trange = ''  # select the time range for imaging, leave it blank for using the entire time interval in the data
twidth = 2  # make one image out of every 2 time integrations
xycen = [380., 50.]  # define the center of the output map, in solar X and Y. Unit: arcsec
xran = [340., 420.]  # plot range in solar X. Unit: arcsec
yran = [10., 90.]  # plot range in solar Y. Unit: arcsec
antennas = ''  # default is to use all 13 EOVSA 2-m antennas. 
npix = 128  # number of pixels in the image
cell = '2arcsec'  # pixel scale in arcsec
pol = 'XX'  # polarization to image, use XX for now
pbcor = True  # correct for primary beam response?
grid_spacing = 2. * u.deg  # spacing for plotting longitude and latitude grid in degrees
outdir = './image_series/'  # Specify where you want to save the output fits files
imresfile = 'imres.npz'  # File to write the imageing results summary
outimgpre = 'EO'  # Something to add to the image name
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

############# This block checks if the script is being run on Virgo ############
import socket

if socket.gethostname() == 'ip-172-26-5-203.ec2.internal':
    print('!!!!!!Caution!!!!!!: We detected that you are trying to run this computationally intensive script on Virgo.')
    print('Please do not try to run this script when Virgo is busy (e.g., during the tutorial)')
    msg0 = 'Do you wish to proceed?'
    if sys.version_info[0] < 3:
        shall0 = raw_input("%s (y/N) " % msg0).lower() == 'y'
    else:
        shall0 = input("%s (y/N) " % msg0).lower() == 'y'
    if shall0:
        print('Current CPU Load: {:.0f}%'.format(psutil.cpu_percent()))
        print('Current Memory Usage: {:.0f}%'.format(psutil.virtual_memory()[2]))
        msg1 = 'Do you still wish to proceed?'
        if sys.version_info[0] < 3:
            shall1 = raw_input("%s (y/N) " % msg1).lower() == 'y'
        else:
            shall1 = input("%s (y/N) " % msg1).lower() == 'y'
        if shall1:
            print('You win. Continue running on Virgo...')
        else:
            sys.exit("Abandon Ship...")
    else:
        sys.exit("Abandon Ship...")


#################################################################################

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


################### MAIN BLOCK FOR CLEAN  ##################
if not os.path.exists(outdir):
    os.makedirs(outdir)
imname0 = outdir + '/' + outimgpre
# Covert in put solar XY (xycen) to new phasecenter in RA and DEC 
try:
    phasecenter, tmid = hf.calc_phasecenter_from_solxy(vis, timerange=trange, xycen=xycen)
    print('use phasecenter: ' + phasecenter)
except:
    print('Provided time format not recognized by astropy.time.Time')
    print('Please use format trange="yyyy/mm/dd/hh:mm:ss~yyyy/mm/dd/hh:mm:ss"')
# Start to clean
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
                  niter=200,
                  gain=0.1,
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
    clnres[s]['FreqGHz'] = [cfreqs[int(sp)] / 1.e9] * len(clnres[s]['ImageName'])

########### Combine ALL FITS IMAGES INTO MULTI-FREQUENCY IMAGE CUBES ###########
ntime = len(clnres[0]['ImageName'])
import suncasa.utils.fits_wrap as fw

imres = {}
for k in ['EndTime', 'BeginTime']:
    imres[k] = clnres[0][k]

imres['FreqGHz'] = [clnres[s]['FreqGHz'][0] for s in range(len(clnres))]
imres['ImageName'] = []
imres['Succeeded'] = np.vstack([clnres[s]['Succeeded'] for s in range(len(clnres))])

for i in range(ntime):
    fitsfile = [clnres[s]['ImageName'][i] for s in range(len(clnres))]
    imname = clnres[0]['ImageName'][i]
    imname_ = imname.replace(imname[-8:-5], 'ALLBD')
    imres['ImageName'].append(imname_)
    fw.fits_wrap_spwX(fitsfile, outfitsfile=imname_)
    for s in fitsfile:
        os.system('rm -rf {}'.format(s))

# imageing results summary are saved to "imresfile"
np.savez(outdir + imresfile, imres=imres)
#################################################################################


########### PLOT ALL THE FITS IMAGES USING SUNPY.MAP ###########
imres = np.load(outdir + imresfile, encoding="latin1")['imres'].item()
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
if os.path.exists(specfile):
    specdata = np.load(specfile)
else:
    from suncasa.utils import dspec as ds

    uvrange = '0.1~1.0km'
    ds.get_dspec(vis, specfile=specfile, uvrange=uvrange, domedian=True)
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
tim_axvspan = ax.axvspan(Time(imres['BeginTime'][tidx]).plot_date, Time(imres['EndTime'][tidx]).plot_date,
                         facecolor='w', alpha=0.6)
ax.set_xlabel('Start Time ({})'.format(spec_tim[tidx].datetime.strftime('%b-%d-%Y %H:%M:%S')))
## get the spectrum at the peak time
tidxmax = np.nanargmax(np.nanmean(spec_plt, axis=0))
peak_frm = np.nanargmin(np.abs(Time(imres['BeginTime']).plot_date - spec_tim[tidxmax].plot_date))
fsfile = imres['ImageName'][peak_frm]
if os.path.exists(fsfile):
    ## get the dmaxs for images plot from the image at peak time
    hdu = fits.open(fsfile)
    dmaxs = [np.nanmax(hdu[0].data[0, s, :, :]) for s in range(len(spws))]
    hdu.close()
else:
    ## get the dmaxs for images plot from the dynamic spectrum
    spec_v = medfilt(spec_plt[:, tidxmax], len(spec[:, 0]) / 100 * 2 + 1)
    spec_v_ = np.interp(cfreqs, freqghz, spec_v) / 1e4  ## flux density in sfu
    dmaxs = sfu2tb(cfreqs * 1e9, spec_v_, bmsz)

plt_ax = True
for tidx in tqdm(range(len(imres['ImageName']))):
    fsfile = imres['ImageName'][tidx]
    if os.path.exists(fsfile):
        hdu = fits.open(fsfile)
    else:
        data = np.zeros(imsize)
    for s, sp in enumerate(spws):
        cfreq = imres['FreqGHz'][s]
        ax = axs[s]
        if os.path.exists(fsfile):
            data = hdu[0].data[0, s, :, :].reshape(imsize)
            data[np.isnan(data)] = 0.0
            eomap = smap.Map(data, hdu[0].header)
            if tidx == 0 or plt_ax:
                eomap_ = pmX.Sunmap(eomap)
                im.append(eomap_.imshow(axes=ax, vmax=dmaxs[s], vmin=dmaxs[s] * 0.05))
                eomap_.draw_limb(axes=ax)
                eomap_.draw_grid(axes=ax, grid_spacing=grid_spacing)
            else:
                im[s].set_array(data)
            if tidx == 0:
                ax.text(0.95, 0.95, '{0:.1f} GHz'.format(cfreq), transform=ax.transAxes, ha='right', va='top',
                        color='w', fontweight='bold')
        else:
            if plt_ax:
                pass
            else:
                im[s].set_array(data)

    if plt_ax:
        for s, sp in enumerate(spws):
            ax = axs[s]
            ax.set_title(' ')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(xran)
            ax.set_ylim(yran)
    if os.path.exists(fsfile):
        hdu.close()
        plt_ax = False

    if tidx == 0:
        ax = axs[0]
        timetext = ax.text(0.95, 0.02, eomap.date.strftime('%H:%M:%S'), ha='right', va='bottom',
                           transform=ax.transAxes, color='w', fontweight='bold')
    else:
        timetext.set_text(eomap.date.strftime('%H:%M:%S'))

    tim_axvspan_xy = tim_axvspan.get_xy()
    tim_axvspan_xy[np.array([0, 1, 4]), 0] = Time(imres['BeginTime'][tidx]).plot_date
    tim_axvspan_xy[np.array([2, 3]), 0] = Time(imres['EndTime'][tidx]).plot_date
    tim_axvspan.set_xy(tim_axvspan_xy)
    # tim_axvspan.set_xy(tim_axvspan_xy)
    figname = fsfile[:-9] + '.png'
    fig.savefig(figname, dpi=100)

plt.ion()

DButil.img2html_movie('{}/EO'.format(outdir))
