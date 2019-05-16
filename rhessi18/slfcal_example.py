from suncasa.utils import helioimage2fits as hf
import os
import numpy as np
import pickle
from matplotlib import gridspec as gridspec
from sunpy import map as smap
from matplotlib import pyplot as plt
import time
'''
Example script for self-calibrating EOVSA flare data
'''
# History:
#   2019-May-15 BC
#       Created a new example script based on B. Chen's practice for self-calibrating 
#       the 2017 Aug 21 20:20 UT flare data. Made it available for EOVSA tutorial at 
#       RHESSI XVIII Workshop (http://rhessi18.umn.edu/)

# =========== task handlers =============
dofullsun=0 # initial full-sun imaging
domasks=0 # get masks
doslfcal=0 # main cycle of doing selfcalibration
doapply=1 # apply the results
doclean_slfcaled=1 # perform clean for self-calibrated data

# ============ declaring the working directories ============
workdir = os.getcwd()+'/' #main working directory. Using current directory in this example
slfcaldir = workdir+'slfcal/' #place to put all selfcalibration products
imagedir = slfcaldir+'images/' #place to put all selfcalibration images
maskdir = slfcaldir+'masks/' #place to put clean masks
imagedir_slfcaled = slfcaldir+'images_slfcaled/' #place to put final self-calibrated images
caltbdir = slfcaldir+'caltbs/' # place to put calibration tables
# make these directories if they do not already exist
dirs = [workdir, slfcaldir, imagedir, maskdir, imagedir_slfcaled, caltbdir]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# ============ Split a short time for self-calibration ===========
# input visibility
ms_in = workdir + 'IDB20170821201800-202300.4s.corrected.ms'
# output, selfcaled, visibility
ms_slfcaled = workdir + os.path.basename(ms_in).replace('corrected','slfcaled') 
# intermediate small visibility for selfcalbration 
# selected time range for generating self-calibration solutions
trange='2017/08/21/20:21:10~2017/08/21/20:21:30'
slfcalms = slfcaldir+'slfcalms.XX.slfcal'
slfcaledms = slfcaldir+'slfcalms.XX.slfcaled'
if not os.path.exists(slfcalms):
    split(vis=ms_in,outputvis=slfcalms,datacolumn='data',timerange=trange,correlation='XX')

# ============ Prior definitions for spectral windows, antennas, pixel numbers =========
spws=[str(s+1) for s in range(30)]
antennas='0~12' 
npix=512
nround=3 #number of slfcal cycles
# parameters specific to the event (found from step 1)
phasecenter='J2000 10h02m59 11d58m14'
xran=[280,480]
yran=[-50,150]

# =========== Step 1, doing a full-Sun image to find out phasecenter and appropriate field of view =========
if dofullsun:
    #initial mfs clean to find out the image phase center
    im_init='fullsun_init'
    os.system('rm -rf '+im_init+'*')
    tclean(vis=slfcalms,
            antenna='0~12',
            imagename=im_init,
            spw='1~15',
            specmode='mfs',
            timerange=trange,
            imsize=[npix],
            cell=['5arcsec'],
            niter=1000,
            gain=0.05,
            stokes='I',
            restoringbeam=['30arcsec'],
            interactive=False,
            pbcor=True)

    hf.imreg(vis=slfcalms,imagefile=im_init+'.image.pbcor',fitsfile=im_init+'.fits',
             timerange=trange,usephacenter=False,verbose=True)
    clnjunks = ['.flux', '.mask', '.model', '.psf', '.residual','.sumwt','.pb','.image']
    for clnjunk in clnjunks:
        if os.path.exists(im_init + clnjunk):
            os.system('rm -rf '+im_init + clnjunk)

    from sunpy import map as smap
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    eomap=smap.Map(im_init+'.fits')
    #eomap.data=eomap.data.reshape((npix,npix))
    eomap.plot_settings['cmap'] = plt.get_cmap('jet')
    eomap.plot(axes = ax)
    eomap.draw_limb()
    plt.show()
    viewer(im_init+'.image.pbcor')

# =========== Step 2 (optional), generate masks =========
# if skipped, will not use any masks
if domasks:
    clearcal(slfcalms)
    delmod(slfcalms)
    antennas='0~12' 
    pol='XX'
    imgprefix=maskdir+'slf_t0'

    # step 1: set up the clean masks
    img_init=imgprefix+'_init_ar_'
    os.system('rm -rf '+img_init+'*')
    #spwrans_mask=['1~5','6~12','13~20','21~30']
    spwrans_mask=['1~12']
    #convert to a list of spws
    spwrans_mask_list = [[str(i) for i in (np.arange(int(m.split('~')[0]),int(m.split('~')[1])))] for m in spwrans_mask]
    masks=[]
    imnames=[]
    for spwran in spwrans_mask:
        imname=img_init+spwran.replace('~','-')
        try:
            tclean(vis=slfcalms,
                    antenna='0~12',
                    imagename=imname,
                    spw=spwran,
                    specmode='mfs',
                    timerange=trange,
                    imsize=[npix],
                    cell=['2arcsec'],
                    niter=1000,
                    gain=0.05,
                    stokes='XX',
                    restoringbeam=['20arcsec'],
                    phasecenter=phasecenter,
                    weighting='briggs',
                    robust=1.0,
                    interactive=True,
		    datacolumn='data',
                    pbcor=True,
                    savemodel='modelcolumn')
            imnames.append(imname+'.image')
            masks.append(imname+'.mask')
            clnjunks = ['.flux', '.model', '.psf', '.residual']
            for clnjunk in clnjunks:
                if os.path.exists(imname + clnjunk):
                    os.system('rm -rf '+ imname + clnjunk)
        except:
            print('error in cleaning spw: '+spwran)

    pickle.dump(masks,open(slfcaldir+'masks.p','wb'))

# =========== Step 3, main step of selfcalibration =========
if doslfcal:
    if os.path.exists(slfcaldir+'masks.p'):
        masks=pickle.load(open(slfcaldir+'masks.p','rb'))
    if not os.path.exists(slfcaldir+'masks.p'):
        print 'masks do not exist. Use default mask'
        masks=[]
    os.system('rm -rf '+imagedir+'*')
    os.system('rm -rf '+caltbdir+'*')
    #first step: make a mock caltable for the entire database
    print('Processing ' + trange)
    slftbs=[]
    calprefix=caltbdir+'slf'
    imgprefix=imagedir+'slf'
    tb.open(slfcalms+'/SPECTRAL_WINDOW')
    reffreqs=tb.getcol('REF_FREQUENCY')
    bdwds=tb.getcol('TOTAL_BANDWIDTH')
    cfreqs=reffreqs+bdwds/2.
    tb.close()
    # starting beam size at 3.4 GHz in arcsec
    sbeam=40.
    strtmp=[m.replace(':','') for m in trange.split('~')]
    timestr='t'+strtmp[0]+'-'+strtmp[1]
    refantenna='0'
    # number of iterations for each round
    niters=[100, 300, 500]
    # roburst value for weighting the baselines
    robusts=[1.0, 0.5, 0.0]
    # apply calibration tables? Set to true for most cases
    doapplycal=[1,1,1]
    # modes for calibration, 'p' for phase-only, 'a' for amplitude only, 'ap' for both
    calmodes=['p','p','a']
    # setting uvranges for model image (optional, not used here)
    uvranges=['','',''] 
    for n in range(nround):
        slfcal_tb_g= calprefix+'.G'+str(n)
        fig = plt.figure(figsize=(8.4,7.))
        gs = gridspec.GridSpec(5, 6)
        for s,sp in enumerate(spws):
            print 'processing spw: '+sp
            cfreq=cfreqs[int(sp)]
            # setting restoring beam size (not very useful for selfcal anyway, but just to see the results)
            bm=max(sbeam*cfreqs[1]/cfreq, 6.)
            slfcal_img = imgprefix+'.spw'+sp.zfill(2)+'.slfcal'+str(n)
            # only the first round uses nearby spws for getting initial model
            if n == 0:
                spbg=max(int(sp)-2,1)
                sped=min(int(sp)+2,30)
                spwran=str(spbg)+'~'+str(sped)
                print('using spw {0:s} as model'.format(spwran))
                if 'spwrans_mask' in vars():
                    for m, spwran_mask in enumerate(spwrans_mask):
                        if sp in spwran_mask:
                            mask = masks[m]
                            print('using mask {0:s}'.format(mask))
                            findmask = True
                    if not findmask:
                        print('mask not found. Do use any masks')
            else:
                spwran = sp
                if 'spwrans_mask' in vars():
                    for m, spwran_mask in enumerate(spwrans_mask):
                        if sp in spwran_mask:
                            mask = masks[m]
                            print 'using mask {0:s}'.format(mask)
                            findmask = True
                    if not findmask:
                        print('mask not found. Do use any masks')
            try:
                tclean(vis=slfcalms,
                        antenna=antennas,
                        imagename=slfcal_img,
                        uvrange=uvranges[n],
                        spw=spwran,
                        specmode='mfs',
                        timerange=trange,
                        imsize=[npix],
                        cell=['2arcsec'],
                        niter=niters[n],
                        gain=0.05,
                        stokes='XX', #use pol XX image as the model
                        weighting='briggs',
                        robust=robusts[n],
                        phasecenter=phasecenter,
                        mask=mask,
                        restoringbeam=[str(bm)+'arcsec'],
                        pbcor=False,
                        interactive=False,
                        savemodel='modelcolumn')
                if os.path.exists(slfcal_img+'.image'):
                    fitsfile=slfcal_img+'.fits'
                    hf.imreg(vis=slfcalms,imagefile=slfcal_img+'.image',fitsfile=fitsfile,
                             timerange=trange,usephacenter=False,toTb=True,verbose=False,overwrite=True)
                clnjunks = ['.mask','.flux', '.model', '.psf', '.residual', '.image','.pb','.image.pbcor','.sumwt']
                for clnjunk in clnjunks:
                    if os.path.exists(slfcal_img + clnjunk):
                        os.system('rm -rf '+ slfcal_img + clnjunk)
                ax = fig.add_subplot(gs[s])
                eomap=smap.Map(fitsfile)
                eomap.plot_settings['cmap'] = plt.get_cmap('jet')
                eomap.plot(axes = ax)
                eomap.draw_limb()
                #eomap.draw_grid()
                ax.set_title(' ')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
                ax.set_xlim(xran)
                ax.set_ylim(yran)
                os.system('rm -f '+ fitsfile)

            except:
                print 'error in cleaning spw: '+sp
                print 'using nearby spws for initial model'
                sp_e=int(sp)+2
                sp_i=int(sp)-2
                if sp_i < 1:
                    sp_i = 1
                if sp_e > 30:
                    sp_e = 30
                sp_=str(sp_i)+'~'+str(sp_e)
                try:
                    tget(tclean)
                    spw=sp_
                    print('using spw {0:s} as model'.format(sp_))
                    tclean()
                except:
                    print 'still not successful. abort...'
                    break

            gaincal(vis=slfcalms, refant=refantenna,antenna=antennas,caltable=slfcal_tb_g,spw=sp, uvrange='',\
                    gaintable=[],selectdata=True,timerange=trange,solint='inf',gaintype='G',calmode=calmodes[n],\
                    combine='',minblperant=4,minsnr=2,append=True)
            if not os.path.exists(slfcal_tb_g):
                print 'No solution found in spw: '+sp
        figname=imagedir+'slf_t0_n{:d}.png'.format(n)
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(figname)
        time.sleep(10)
        plt.close()

        if os.path.exists(slfcal_tb_g):
            slftbs.append(slfcal_tb_g)
            slftb=[slfcal_tb_g]
            os.chdir(slfcaldir)
            if calmodes[n] == 'p': 
                plotcal(caltable=slfcal_tb_g,antenna='1~12',xaxis='freq',yaxis='phase',\
                        subplot=431,plotrange=[-1,-1,-180,180],iteration='antenna',figfile=slfcal_tb_g+'.png',showgui=False)
            if calmodes[n] == 'a':
                plotcal(caltable=slfcal_tb_g,antenna='1~12',xaxis='freq',yaxis='amp',\
                        subplot=431,plotrange=[-1,-1,0,2.],iteration='antenna',figfile=slfcal_tb_g+'.png',showgui=False)
            os.chdir(workdir)


        if doapplycal[n]:
            clearcal(slfcalms)
            delmod(slfcalms)
            applycal(vis=slfcalms,gaintable=slftb,spw=','.join(spws),selectdata=True,\
                     antenna=antennas,interp='nearest',flagbackup=False,applymode='calonly',calwt=False)

        if n < nround-1: 
            prompt=raw_input('Continuing to selfcal?')
            #prompt='y'
            if prompt.lower() == 'n':
                if os.path.exists(slfcaledms):
                    os.system('rm -rf '+slfcaledms)
                split(slfcalms,slfcaledms,datacolumn='corrected')
                print 'Final calibrated ms is {0:s}'.format(slfcaledms)
                break
            if prompt.lower() == 'y':
                slfcalms_=slfcalms+str(n)
                if os.path.exists(slfcalms_):
                    os.system('rm -rf '+slfcalms_)
                split(slfcalms,slfcalms_,datacolumn='corrected')
                slfcalms=slfcalms_
        else:
            if os.path.exists(slfcaledms):
                os.system('rm -rf '+slfcaledms)
            split(slfcalms,slfcaledms,datacolumn='corrected')
            print 'Final calibrated ms is {0:s}'.format(slfcaledms)

# =========== Step 4: Apply self-calibration tables =========
if doapply:
    import glob
    os.chdir(workdir)
    clearcal(ms_in)
    clearcal(slfcalms)
    applycal(vis=slfcalms,gaintable=slftbs,spw=','.join(spws),selectdata=True,\
             antenna=antennas,interp='linear',flagbackup=False,applymode='calonly',calwt=False)
    applycal(vis=ms_in,gaintable=slftbs,spw=','.join(spws),selectdata=True,\
             antenna=antennas,interp='linear',flagbackup=False,applymode='calonly',calwt=False)
    if os.path.exists(ms_slfcaled):
        os.system('rm -rf '+ms_slfcaled)
    split(ms_in, ms_slfcaled,datacolumn='corrected')

# =========== Step 5: Generate final self-calibrated images (optional) =========
if doclean_slfcaled:
    import glob
    pol='XX'
    print('Processing ' + trange)
    img_final=imagedir_slfcaled+'/slf_final_{0}_t0'.format(pol)
    vis = ms_slfcaled
    spws=[str(s+1) for s in range(30)]
    tb.open(vis+'/SPECTRAL_WINDOW')
    reffreqs=tb.getcol('REF_FREQUENCY')
    bdwds=tb.getcol('TOTAL_BANDWIDTH')
    cfreqs=reffreqs+bdwds/2.
    tb.close()
    sbeam=30.
    from matplotlib import gridspec as gridspec
    from sunpy import map as smap
    from matplotlib import pyplot as plt
    fitsfiles=[]
    for s,sp in enumerate(spws):
        cfreq=cfreqs[int(sp)]
        bm=max(sbeam*cfreqs[1]/cfreq,4.)
        imname=img_final+'_s'+sp.zfill(2)
        fitsfile=imname+'.fits'
        if not os.path.exists(fitsfile):
            print 'cleaning spw {0:s} with beam size {1:.1f}"'.format(sp,bm)
            try:
                tclean(vis=vis,
                        antenna=antennas,
                        imagename=imname,
                        spw=sp,
                        specmode='mfs',
                        timerange=trange,
                        imsize=[npix],
                        cell=['1arcsec'],
                        niter=1000,
                        gain=0.05,
                        stokes=pol,
                        weighting='briggs',
                        robust=2.0,
                        restoringbeam=[str(bm)+'arcsec'],
                        phasecenter=phasecenter,
                        mask='',
                        pbcor=True,
                        interactive=False)
            except:
                print 'cleaning spw '+sp+' unsuccessful. Proceed to next spw'
                continue
            if os.path.exists(imname+'.image.pbcor'):
                imn = imname+'.image.pbcor'
                hf.imreg(vis=vis,imagefile=imn,fitsfile=fitsfile,
                         timerange=trange,usephacenter=False,toTb=True,verbose=False)
            fitsfiles.append(fitsfile)
            junks=['.flux','.model','.psf','.residual','.mask','.image','.pb','.image.pbcor','.sumwt']
            for junk in junks:
                if os.path.exists(imname+junk):
                    os.system('rm -rf '+imname+junk)
        else:
            print('fits file '+fitsfile+' already exists, skip clean...')
            fitsfiles.append(fitsfile)

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
        plt.text(0.98,0.85,'{0:.1f} GHz'.format(cfreq/1e9),transform=ax.transAxes,ha='right',color='w',fontweight='bold')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()

