from suncasa.utils import helioimage2fits as hf
import os
import numpy as np
from matplotlib import gridspec as gridspec
from sunpy import map as smap
from matplotlib import pyplot as plt
from astropy.time import Time
'''
Example script for doing a 30-band spectral imaging at a given time
'''
# History:
#   2019-May-17 BC
#       Created a new example script based on B. Chen's practice for spectral imaging
#       the 2017 Aug 21 20:20 UT flare data. Made it available for EOVSA tutorial at 
#       RHESSI XVIII Workshop (http://rhessi18.umn.edu/)

################### USER INPUT GOES IN THIS BLOK ########################
vis = 'IDB20170821201800-202300.4s.slfcaled.ms' # input visibility data
trange = '2017/08/21/20:21:00~2017/08/21/20:21:30' #select the time range for imaging (averaging)
xycen = [380., 50.] # define the center of the output map, in solar X and Y. Unit: arcsec
xran = [280., 480.] # plot range in solar X. Unit: arcsec
yran = [-50., 150.] # plot range in solar Y. Unit: arcsec
antennas = '0~12' # use all 13 EOVSA antennas. If some antenna is no good, drop it in this selection
npix = 256 # number of pixels in the image
cell = '1arcsec' # pixel scale in arcsec
pol = 'XX' # polarization to image, use XX for now
pbcor = True # correct for primary beam response?
outdir = './images' # Specify where you want to save the output fits files
if not os.path.exists(outdir):
    os.makedirs(outdir)
outimgpre = 'EO' # Something to add to the image name
#################################################################################

################### CHANGE ONLY IF YOU KNOW WHAT YOU ARE DOING ##################
# convert xycen to phase center in RA and DEC
(btstr, etstr) = trange.split('~')
try: 
    bt = Time(qa.time(btstr,form='fits')[0]).mjd
    et = Time(qa.time(etstr,form='fits')[0]).mjd
    midt = (bt + et) / 2.
    midtstr = ((Time(midt,format='mjd').isot).replace(':','')).replace('-','')
except:
    print('Provided time format not recognized by astropy.time.Time')
    print('Please use format trange="yyyy/mm/dd/hh:mm:ss~yyyy/mm/dd/hh:mm:ss"')
# read ephemeris from JPL Horizons
eph = hf.read_horizons(t0=Time(midt, format='mjd'))
x0 = np.radians(xycen[0] / 3600.)
y0 = np.radians(xycen[1] / 3600.)
p0 = np.radians(eph['p0'][0])  # p angle in radians
raoff = -((x0) * np.cos(p0) - y0 * np.sin(p0)) / np.cos(eph['dec'][0])
decoff = (x0) * np.sin(p0) + y0 * np.cos(p0)
# read RA DEC from visibility data
tb.open(vis + '/FIELD')
phadir = tb.getcol('PHASE_DIR').flatten()
tb.close()
ra0 = phadir[0]
dec0 = phadir[1]
# re-calculate new phasecenter that corresponds to the desired map center in solar X and Y
newra = ra0 + raoff
newdec = dec0 + decoff
phasecenter = 'J2000 ' + str(newra) + 'rad ' + str(newdec) + 'rad'
print('use phasecenter: ' + phasecenter)
#################################################################################

################### (OPTIONAL) DEFINE RESTORING BEAM SIZE  ##################
### (Optional) the following is to define beam size ###
# read frequency information from visibility data
tb.open(vis + '/SPECTRAL_WINDOW')
reffreqs=tb.getcol('REF_FREQUENCY')
bdwds=tb.getcol('TOTAL_BANDWIDTH')
cfreqs=reffreqs+bdwds/2.
tb.close()
spws=[str(s+1) for s in range(30)]
sbeam=30.
bmsz=[max(sbeam*cfreqs[1]/f,6.) for f in cfreqs] 
#################################################################################


################### MAIN BLOCK FOR CLEAN  ##################
imname0 = outdir + '/'+ outimgpre 
fitsfiles=[]
for s,sp in enumerate(spws):
    cfreq=cfreqs[int(sp)]
    imname = imname0 + midtstr + '_S' + sp.zfill(2) # add band information to image name
    fitsfile = imname + '.fits'
    if not os.path.exists(fitsfile):
        print 'cleaning spw {0:s} with beam size {1:.1f}"'.format(sp,bmsz[s])
        try:
            tclean(vis=vis,
                    antenna=antennas,
                    imagename=imname,
                    spw=sp,
                    specmode='mfs',
                    timerange=trange,
                    imsize=[npix],
                    cell=[cell],
                    niter=1000,
                    gain=0.05,
                    stokes=pol,
                    weighting='briggs',
                    robust=0.0,
                    restoringbeam=[str(bmsz[s])+'arcsec'],
                    phasecenter=phasecenter,
                    mask='',
                    pbcor=pbcor,
                    interactive=False)
        except:
            print('cleaning spw '+sp+' unsuccessful. Proceed to next spw')
            continue
        if pbcor:
            imn = imname+'.image.pbcor'
        else:
            imn = imname+'.image'
        if os.path.exists(imn):
            hf.imreg(vis=vis,imagefile=imn,fitsfile=fitsfile,
                     timerange=trange,usephacenter=False,toTb=True,verbose=False)
            fitsfiles.append(fitsfile)
        else:
            print('cleaning spw '+sp+' unsuccessful. Proceed to next spw')
        junks=['.flux','.model','.psf','.residual','.mask','.image','.pb','.image.pbcor','.sumwt']
        for junk in junks:
            if os.path.exists(imname+junk):
                os.system('rm -rf '+imname+junk)
    else:
        print('fits file '+fitsfile+' already exists, skip clean...')
        fitsfiles.append(fitsfile)
# all the output fits files are given in "fitsfiles"
#################################################################################

########### PLOT ALL THE FITS IMAGES USING SUNPY.MAP ###########
fig = plt.figure(figsize=(8.4,7.))
gs = gridspec.GridSpec(5, 6)
for s,sp in enumerate(spws):
    cfreq=cfreqs[int(sp)]
    ax = fig.add_subplot(gs[s])
    eomap=smap.Map(fitsfiles[s])
    eomap.plot_settings['cmap'] = plt.get_cmap('jet')
    eomap.plot(axes = ax)
    eomap.draw_limb()
    ax.set_title(' ')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(xran)
    ax.set_ylim(yran)
    plt.text(0.97,0.85,'{0:.1f} GHz'.format(cfreq/1e9),transform=ax.transAxes,ha='right',color='w',fontweight='bold')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()

