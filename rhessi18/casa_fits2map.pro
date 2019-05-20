;+
; Project     : VLA, EOVSA, and possibly ALMA 
;
; Name        : CASA_FITS2MAP
;
; Purpose     : Convert an array of 2-, 3-, or 4-D CASA fits files to
;               a map or 1-, 2-, or 3-D array of ssw map structure
;
; Category    :
;
; Syntax      : casa_fits2map, files, maps
;
; Dependencies: casa_readfits.pro 
; Inputs      : files = a 2-, 3-, or 4-D CASA fits file or a temporal array of CASA fits files
;               the input fits files are organized in X, Y, frequency, and polarization
;
; Output      : maps = array of maps.  If 3-D array of maps, organization is frequency, polarization, 
;               time
;
; Keywords    : calrms
;                   If set, calculate the rms and snr of the map from
;                   - the entire map, if neither rmsxran nor rmsyran is set
;                   - Part of the map, if rmsxran and/or rmsyran are set
;               rmsxran
;                   x range to calculate rms. If not set, use x range of the whole map
;               rmsyran
;                   y range to calculate rms. If not set, use y range of the whole map
;
;
; History     : Written by Bin Chen (bin.chen@njit.edu) based on hsi_fits2map.pro by P. T. Gallagher
;               Originally called vla_fits2map.pro
;
;               16-Feb-2016, Bin Chen - set roll_angle to zero for two-dimensional maps. Otherwise it takes -p_angle 
;               19-May-2019, Bin Chen - added additional keywords to the map structure, and renamed to casa_fits2map.pro
;                               * calcrms: whether or not to calculate rms, which uses rmsxran and rmsyran
;                               * and a function to calculate rms
;
; Contact     : bin.chen@njit.edu
;-
function calc_rms, map, rmsxran=rmsxran, rmsyran=rmsyran, snr=snr
; calculate rms and snr of a given map. Only one dimensional map is accepted
    if keyword_set(rmsxran) or keyword_set(rmsyran) then begin
        if not keyword_set(rmsxran) then rmsxran=get_map_xrange(map[0])
        if not keyword_set(rmsyran) then rmsyran=get_map_yrange(map[0])
        rmsxran=float(rmsxran) & rmsyran=float(rmsyran)
        sub_map, map, smap, xrange=rmsxran, yrange=rmsyran
    endif else begin
        ; use entire map
        rmsxran=get_map_xrange(map[0])
        rmsyran=get_map_yrange(map[0])
        smap = map
    endelse
    ;select only pixels smaller than 10% of the maximum
    idx = where(smap.data lt 0.1*max(map.data,/nan))
    rms = sqrt((moment(smap.data[idx],/nan))[1])
    snr = max(map.data,/nan)/rms
    return, rms
end

pro casa_fits2map, files, maps, calcrms=calcrms, rmsxran=rmsxran, rmsyran=rmsyran

casa_readfits,files,index,data,/silent
n_dim = size( data, /n_dim )
;map stokes values in FITS to actual keys, reference: Greisen & Calabretta 2002, A&A, 395, 1061
stokesvals=[1,2,3,4,-1,-2,-3,-4,-5,-6,-7,-8]
stokeskeys=['I','Q','U','V','RR','LL','RL','LR','XX','YY','XY','YX']
case 1 of
    n_dim eq 2: begin
        index2map, index, data, map
        ;determine the frequency axis
        if (tag_exist(index,'crval3') and index.ctype3 eq 'FREQ') then begin 
            if tag_exist(index,'cunit3') and index.cunit3 eq 'Hz' then begin
                freq=index.crval3/1e9
                frequnit='GHz'
            endif
        endif
        if (tag_exist(index,'crval4') and index.ctype4 eq 'FREQ') then begin
            if tag_exist(index,'cunit4') and index.cunit4 eq 'Hz' then begin
                freq=index.crval4/1e9
                frequnit='GHz'
            endif
        endif
        ;determine the stokes axis
        if (tag_exist(index,'crval3') and index.ctype3 eq 'STOKES') then $
            stokesnum=index.crval3
        if (tag_exist(index,'crval4') and index.ctype4 eq 'STOKES') then $
            stokesnum=index.crval4
        ;check if the stokes value exist in the dictionary
        sind=where(stokesvals eq stokesnum,nsind)
        if nsind eq 1 then begin
            stokes=stokeskeys[sind[0]]
        endif else begin
            print, 'error when finding the stokes parameter!'
            return
        endelse

        add_prop, map, id = index.telescop  + ' ' + stokes + ' ' + $
            strtrim(string(freq,format='(f6.3)'),2) + ' ' + frequnit
        add_prop, map, freq = freq
        add_prop, map, frequnit = 'GHz' 
        add_prop, map, stokes = stokes
        add_prop, map, dataunit = index.bunit 
        add_prop, map, datatype = index.btype 
        add_prop, map, bmaj = index.bmaj
        add_prop, map, bmin = index.bmin
        add_prop, map, bpa = index.bpa
        add_prop, map, rsun = index.rsun_obs
        add_prop, map, l0 = index.HGLN_OBS
        add_prop, map, b0 = index.HGLT_OBS
        add_prop, map, comment = 'Converted by CASA_FITS2MAP.PRO'
        map.roll_angle=0.
        if keyword_set(calcrms) then begin
            rms = calc_rms(map, rmsxran=rmsxran, rmsyran=rmsyran, snr=snr)
            add_prop, map, rms = rms
            add_prop, map, snr = snr
            add_prop, map, rmsunit = index.bunit
            add_prop, map, rmsxran = rmsxran
            add_prop, map, rmsyran = rmsyran
        endif
        maps = map
    end

    n_dim eq 3: begin
        ; determine the frequency axis
        index0=index[0]
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit3') and index0.cunit3 eq 'Hz' then begin
                freqs=index0.crval3/1e9+findgen(index0.naxis3)*index0.cdelt3/1e9
                frequnit='GHz'
            endif
        endif
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit4') and index0.cunit4 eq 'Hz' then begin
                freqs=index0.crval4/1e9+findgen(index0.naxis4)*index0.cdelt4/1e9
                frequnit='GHz'
            endif
        endif
        nfreqs = n_elements(freqs)
        ; determine the time axis
        time = index0.date_obs
        ; determine the stokes axis
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'STOKES') then $
            stokesnum=index0.crval3
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'STOKES') then $
            stokesnum=index0.crval4
        ;check if the stokes value exist in the dictionary
        sind=where(stokesvals eq stokesnum,nsind)
        if nsind eq 1 then begin
            stokes=stokeskeys[sind[0]]
        endif else begin
            print, 'error when finding the stokes parameter!'
            return
        endelse

        for i = 0, nfreqs - 1 do begin
             freq = freqs[i]
             ;compute xcen and ycen
             ind=index[i]
             get_fits_cen,ind,xcen,ycen
             map = make_map( data( *, *, i ), $
                       xc     = xcen, $
                       yc     = ycen, $
                       dx     = ind.cdelt1, $
                       dy     = ind.cdelt2, $
                       time   = anytim(time,/vms), $
                       id     = ind.telescop + ' ' + stokes + ' '+ $
                                strtrim(string(freq,format='(f6.3)'),2) + ' ' + frequnit , $
                       dur    = ind.exptime, $
                       freq  = freq , $
                       frequnit = 'GHz', $
                       xunits = 'arcsec', $
                       yunits = 'arcsec', $
                       roll_angle = 0., $
                       stokes = stokes)
                add_prop, map, dataunit = ind.bunit 
                add_prop, map, datatype = ind.btype 
                add_prop, map, bmaj = ind.bmaj
                add_prop, map, bmin = ind.bmin
                add_prop, map, bpa = ind.bpa
                add_prop, map, rsun = ind.rsun_obs
                add_prop, map, l0 = ind.HGLN_OBS
                add_prop, map, b0 = ind.HGLT_OBS
                add_prop, map, comment = 'Converted by CASA_FITS2MAP.PRO'
                if keyword_set(calcrms) then begin
                    rms = calc_rms(map, rmsxran=rmsxran, rmsyran=rmsyran, snr=snr)
                    add_prop, map, rms = rms
                    add_prop, map, snr = snr
                    add_prop, map, rmsunit = ind.bunit
                    add_prop, map, rmsxran = rmsxran
                    add_prop, map, rmsyran = rmsyran
                endif

             if ( i eq 0 )  then begin
                maps = replicate(map, nfreqs)
             endif else begin
                maps[i] = map
             endelse
         endfor
       end

    n_dim eq 4: begin ;the third and fourth axes are frequency and stokes 
        ; determine the frequency axis
        index0=index[0,0]
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit3') and index0.cunit3 eq 'Hz' then begin
                freqs=index0.crval3/1e9+findgen(index0.naxis3)*index0.cdelt3/1e9
                frequnit='GHz'
            endif
        endif
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit4') and index0.cunit4 eq 'Hz' then begin
                freqs=index0.crval4/1e9+findgen(index0.naxis4)*index0.cdelt4/1e9
                frequnit='GHz'
            endif
        endif
        nfreqs = n_elements(freqs)
        ; determine the time axis
        time = index[0,0].date_obs
        ;determine the stokes axis
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'STOKES') then $
            stokesnums=index0.crval3+indgen(index0.naxis3)*index0.cdelt3
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'STOKES') then $
            stokesnums=index0.crval4+indgen(index0.naxis4)*index0.cdelt4
        nstokes=n_elements(stokesnums)
        stokes_=strarr(nstokes)
        ;check if the stokes value exist in the dictionary
        for s=0,nstokes-1 do begin
            stokesnum=stokesnums[s]
            sind=where(stokesvals eq stokesnum,nsind)
            if nsind eq 1 then begin
                stokes_[s]=stokeskeys[sind[0]]
            endif else begin
                print, 'error when finding the stokes parameter!'
                return
            endelse
        endfor
        
        ; Make map for all freqs and stokes
        for i = 0, nfreqs - 1 do begin
            freq = freqs[i]
            for j = 0, nstokes - 1 do begin
                 stokes = stokes_[j]
                 ind=index[i,j]
                 get_fits_cen,ind,xcen,ycen
                 map = make_map( data( *, *, i, j), $
                           xc     = xcen, $
                           yc     = ycen, $
                           dx     = ind.cdelt1, $
                           dy     = ind.cdelt2, $
                           time   = anytim(time,/vms), $
                           id     = ind.telescop + ' ' + stokes + ' '+ $
                                    strtrim(string(freq,format='(f6.3)'),2) + ' ' + frequnit , $
                           dur    = ind.exptime, $
                           freq  = freq , $
                           frequnit = frequnit, $
                           xunits = 'arcsec', $
                           yunits = 'arcsec',$
                           roll_angle = 0., $
                           stokes = stokes)
                add_prop, map, dataunit = ind.bunit 
                add_prop, map, datatype = ind.btype 
                add_prop, map, bmaj = ind.bmaj
                add_prop, map, bmin = ind.bmin
                add_prop, map, bpa = ind.bpa
                add_prop, map, rsun = ind.rsun_obs
                add_prop, map, l0 = ind.HGLN_OBS
                add_prop, map, b0 = ind.HGLT_OBS
                add_prop, map, comment = 'Converted by CASA_FITS2MAP.PRO'
                if keyword_set(calcrms) then begin
                    rms = calc_rms(map, rmsxran=rmsxran, rmsyran=rmsyran, snr=snr)
                    add_prop, map, rms = rms
                    add_prop, map, snr = snr
                    add_prop, map, rmsunit = ind.bunit
                    add_prop, map, rmsxran = rmsxran
                    add_prop, map, rmsyran = rmsyran
                endif
                 if ( i eq 0 and j eq 0)  then begin
                    maps = replicate(map, nfreqs, nstokes)
                 endif else begin
                    maps[i,j] = map
                 endelse
            endfor
        endfor
    end

    n_dim eq 5: begin ;the 3rd, 4th, and 5th axes are frequency, stokes, and time 
        ; determine the frequency axis
        index0=index[0,0,0]
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit3') and index0.cunit3 eq 'Hz' then begin
                freqs=index0.crval3/1e9+findgen(index0.naxis3)*index0.cdelt3/1e9
                frequnit='GHz'
            endif
        endif
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'FREQ') then begin 
            if tag_exist(index0,'cunit4') and index0.cunit4 eq 'Hz' then begin
                freqs=index0.crval4/1e9+findgen(index0.naxis4)*index0.cdelt4/1e9
                frequnit='GHz'
            endif
        endif
        nfreqs = n_elements(freqs)
        ; determine the time axis
        times = reform(index[0,0,*].date_obs)
        timesecs = anytim(times)
        ntimes = n_elements(times)
        ;determine the stokes axis
        if (tag_exist(index0,'crval3') and index0.ctype3 eq 'STOKES') then $
            stokesnums=index0.crval3+indgen(index0.naxis3)*index0.cdelt3
        if (tag_exist(index0,'crval4') and index0.ctype4 eq 'STOKES') then $
            stokesnums=index0.crval4+indgen(index0.naxis4)*index0.cdelt4
        nstokes=n_elements(stokesnums)
        stokes_=strarr(nstokes)
        ;check if the stokes value exist in the dictionary
        for s=0,nstokes-1 do begin
            stokesnum=stokesnums[s]
            sind=where(stokesvals eq stokesnum,nsind)
            if nsind eq 1 then begin
                stokes_[s]=stokeskeys[sind[0]]
            endif else begin
                print, 'error when finding the stokes parameter!'
                return
            endelse
        endfor

        ; Make map for all times, frequencies, and polarizations
        for i = 0, nfreqs-1 do begin
            freq=freqs[i]
            for j = 0, nstokes - 1 do begin
                stokes=stokes_[j]
                for k = 0, ntimes - 1 do begin
                     time=times[k]
                     ind=index[i,j,k]
                     get_fits_cen,ind,xcen,ycen
                     map = make_map( data( *, *, i, j, k), $
                               xc     = xcen, $
                               yc     = ycen, $
                               dx     = ind.cdelt1, $
                               dy     = ind.cdelt2, $
                               time   = anytim(time,/vms), $
                               id     = ind.telescop + ' ' + stokes + ' '+ $
                                        strtrim(string(freq,format='(f6.3)'),2) + ' ' + frequnit , $
                               dur    = ind.exptime, $
                               freq  = freq , $
                               frequnit = 'GHz', $
                               xunits = 'arcsec', $
                               yunits = 'arcsec',$
                               roll_angle = 0., $
                               stokes = stokes)
                    add_prop, map, dataunit = ind.bunit 
                    add_prop, map, datatype = ind.btype 
                    add_prop, map, bmaj = ind.bmaj
                    add_prop, map, bmin = ind.bmin
                    add_prop, map, bpa = ind.bpa
                    add_prop, map, rsun = ind.rsun_obs
                    add_prop, map, l0 = ind.HGLN_OBS
                    add_prop, map, b0 = ind.HGLT_OBS
                    add_prop, map, comment = 'Converted by CASA_FITS2MAP.PRO'
                    if keyword_set(calcrms) then begin
                        rms = calc_rms(map, rmsxran=rmsxran, rmsyran=rmsyran, snr=snr)
                        add_prop, map, rms = rms
                        add_prop, map, snr = snr
                        add_prop, map, rmsunit = ind.bunit
                        add_prop, map, rmsxran = rmsxran
                        add_prop, map, rmsyran = rmsyran
                    endif
                     if ( i eq 0 and j eq 0 and k eq 0)  then begin
                        maps = replicate(map, nfreqs, nstokes, ntimes)
                     endif else begin
                        maps[i,j,k] = map
                     endelse
                endfor
            endfor
        endfor
    end
endcase

end
