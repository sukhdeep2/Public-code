import sys
sys.path.insert(0, '/home/sukhdees/project/code/python_scripts/')
from astropy.coordinates import ICRS, Galactic,SkyCoord
from cov_corr import cov_corr
import pickle
#import cosmology
import warnings
import numpy as np
import healpy as hp
from astropy import units as units
from utils import *
from multiprocessing import Pool,Process

cc=cov_corr()
#cosmo=cosmology.cosmology(h=1,Omega_b=0.046,Omega_dm=0.236,Omega_L=0.718)

def jk_read(dataset='',var=None,njk=100,auto_cross='_cross',bins='_bins',
            home='',corr='_final',skiprows=0,**kwargs):
    data_jk={}
    filename=dataset
    try:
        data_jk['data']=np.genfromtxt(home+filename+bins+auto_cross+'_jk_final.dat',
                                      names=True,skip_header=skiprows)
    except:
        print ('file not found')
    try:
        data_jk['data0']=np.genfromtxt(home+filename+bins+auto_cross+corr+'.dat',
                                       names=True,skip_header=skiprows)
    except:
        print ('file not found')
    for i in np.arange(njk):
        data_jk[i]=np.genfromtxt(home+filename+bins+auto_cross+'_jk'+str(i)+corr+'.dat',names=True,
                                 skip_header=skiprows)

    if var and njk>0:
        cov,corr=cc.dict_cov(dic=data_jk,var=var,stack=1,ignore_keys=['data','data0'])
        cov*=njk-1.
        data_jk['cov']=cov
        data_jk['corr']=corr
    return data_jk

def wl_jk_read(dataset='',auto_cross='_wl',bins='bins',home='',calib_bias=0,
               rand_ratio=0,njk=100,**kwargs):
    data_jk=jk_read(dataset=dataset,home=home,auto_cross=auto_cross,corr='',njk=njk,**kwargs)
    rand_jk=jk_read(dataset=dataset+'_rand',home=home,auto_cross=auto_cross,corr='',
                    njk=njk,**kwargs)
    print ('wl_read: calib bias=',calib_bias)
    boost={}
    for i in data_jk.keys():
        data_jk[i]['3sig']-=rand_jk[i]['3sig']
        if i=='data':
            dat_wt_num,a=cc.dict_stack(dic=data_jk,var='5wt_num')
            rand_wt_num,a=cc.dict_stack(dic=rand_jk,var='5wt_num')
            data_jk[i]['5wt_num']+=np.mean(dat_wt_num,axis=a)
            rand_jk[i]['5wt_num']+=np.mean(rand_wt_num,axis=a)
        boost[i]=(data_jk[i]['5wt_num']*rand_ratio)/rand_jk[i]['5wt_num'] #boost factor
        r=boost[i]*1.66*10**6 *calib_bias #calbi bias and c,G constant factor
        data_jk[i]['3sig']*=r
        data_jk[i]['4sig_err']*=r

    sig,a=cc.dict_stack(dic=data_jk,var='3sig',ignore_keys=['data','data0','cov','corr'])
    data_jk['data']['3sig']=np.mean(sig,a)
    nf=njk-1
    data_jk['data']['4sig_err']=np.sqrt(np.var(sig,a)*nf*nf/njk)

    cov,corr=cc.dict_cov(dic=data_jk,var='3sig',stack=1,ignore_keys=['data','data0'])
    cov*=nf
    data_jk['cov']=cov
    data_jk['corr']=corr
    data_jk['rand_jk']=rand_jk
    data_jk['boost']=boost
    return data_jk

def jk_mean(p={},njk=100):
    if check_empty(p):
        print ('jk-mean: got empty dict')
        return p
    p2={}
    nn=np.arange(njk)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    #print mean
    var=np.var(jk_vals,axis=0,ddof=0)*(njk-1)
    p['jk']=mean
    p['jk_err']=np.sqrt(var)
    return p

def jk_mean_percentile(p={},njk=100):
    if check_empty(p):
        print ('jk-mean: got empty dict')
        return p
    dat=[p[i] for i in np.arange(njk)]
    p['jk']=np.mean(dat,axis=0)
    p['jk_err']=np.percentile(dat,[16,84],axis=0)-p['jk']
    p['jk_err']*=np.sqrt((njk-1.)*(njk-1.)/njk)
    return p

def jk_fit(data_jk={},jk_read_func=[],fit_func=[],
           ignore_keys=['data','data0','cov','corr','jk','jk_err','rp'],**kwargs):
    params={}
    if not data_jk:
        data_jk=jk_read_func(**kwargs)
    for i in data_jk.keys():
        if i in ignore_keys:
            continue
        #if i>10:
         #   break
        params[i]=fit_func(data=data_jk[i],**kwargs)
        #print 'jk_fit:',i,params[i]
    params=jk_mean(p=params)
    print ('jk_fit:jk',params['jk'],params['jk_err'])

    for i in ['data','data0','jk']:
        try:
            params[i]=fit_func(data=data_jk[i],**kwargs)
        except:
            do_nothing=1
            #print 'jk_fit::',i,' not fit'
    return params

def gal_jk(mask=[],dat=[],nside=512,njk1=10,njk2=10):
    lm2=hp.pixelfunc.ud_grade(mask,nside_out=nside,pess=True,order_in='Ring',
                              order_out='Ring',dtype='int')
    mask2=lm2.astype('bool')
    nest=False
    njk=njk1 #np.sqrt(njk_tot)
    x=mask2>0
    nii=np.int64(sum(x)/njk+1)

    jkmap=np.zeros_like(mask2,dtype='float64')-999
    ii=np.int64(np.arange(len(mask2)))
    ii=ii[x]
    for i in np.int64(np.arange(njk)):
        if i==nii%njk:
            nii-=1
        indx=ii[i*nii:(i+1)*nii]
        jkmap[indx]=i

    #njk2=njk_tot/njk
    jkmap2=np.zeros_like(jkmap)-999
    ii=np.int64(np.arange(len(jkmap2)))
    for i in np.int64(np.arange(njk)):
        x=jkmap==i
        ii2=ii[x]
        add_f=0
        if i==0:#to get circle at galactic pole
            nii=np.int64(sum(x)/njk2)
            jkmap2[ii2[:nii]]=0
            ii2=ii2[nii:]
            add_f=1
        theta,phi=hp.pixelfunc.pix2ang(nside,ii2,nest=nest)
        phi_bins=np.percentile(phi,np.linspace(0,100,njk2+1-add_f))
        phi_bins[0]-=0.1
        phi_bins[-1]+=0.1
        for j in np.int64(np.arange(njk2-add_f)):
            xphi=phi>phi_bins[j]
            xphi*=phi<phi_bins[j+1]
            indx=ii2[xphi]
            jkmap2[indx]=i*njk2+j+add_f

    #c = ICRS(ra=dat['RA']*units.degree, dec=dat['DEC']*units.degree)#,unit=(units.degree, units.degree))
    #dat_galactic=c.transform_to(Galactic)
    c=SkyCoord(ra=dat['RA']*units.degree, dec=dat['DEC']*units.degree,frame='icrs')
    dat_galactic=c.galactic
    gpix=hp.pixelfunc.ang2pix(nside=nside,nest=nest,theta=dat_galactic.b.radian*-1+np.pi/2.,
                              phi=dat_galactic.l.radian)
    jkgal=jkmap2[gpix]
    return jkgal

def jk_regions(RA=[],DEC=[],njk=10,nside=512,**kwargs):
    d2r=np.pi/180.
    ra2=(RA)*d2r
    dec2=(DEC-90)*-1.*d2r

    nest=False
    mockpix=hp.ang2pix(nside=nside,theta=dec2,phi=ra2,nest=nest)
    mockmap=np.zeros(12*nside**2)
    for i in mockpix:
        mockmap[i]+=1

    hp_masked=-1.6374999999999999e+30
    x1=np.arange(min(mockpix))
    mockmap[x1]=hp_masked
    x2=np.arange(start=max(mockpix)+1,stop=len(mockmap)-1,step=1)
    mockmap[x2]=hp_masked

    pixang=hp.pix2ang(nside=nside,ipix=np.arange(len(mockmap)),nest=nest)
    x1=pixang[1]<min(ra2)
    x2=pixang[1]>max(ra2)
    x3=pixang[0]<min(dec2)
    x4=pixang[0]>max(dec2)
    mockmap[x1+x2+x3+x4]=hp_masked

    dec_tol=0.005
    ra_tol=dec_tol
    njk=10
    dec_tol=0.005
    ra_tol=dec_tol

    nm=np.arange(len(mockmap))
    x=mockmap>-1000
    nm2=nm[x]
    pixang=hp.pix2ang(nside=nside,ipix=nm2)

    jk=np.zeros_like(nm2)-1

    nexp_dec=len(nm2)/njk

    dec_step=(max(dec2)-min(dec2))/(njk*100)
    ra_step=(max(ra2)-min(ra2))/(njk*100)

    dec_lim=np.zeros(njk+1)
    dec_lim[0]=np.min(dec2)
    dec_lim[-1]=np.max(dec2)
    nsum_dec=0

    ra_lim_all={}
    nsum_ra_all={}

    ijk=0
    x1=np.zeros(len(nm2),dtype='bool')
    x2=np.zeros(len(nm2),dtype='bool')
    x3=np.zeros(len(nm2),dtype='bool')
    x4=np.zeros(len(nm2),dtype='bool')
    for i in np.arange(njk):
        dec_min=dec_lim[i]
        j=0
        n=0
        x1*=False
        x1+=pixang[0]>dec_min
        j_step=1
        while n<nexp_dec*0.995:
            x2*=False
            dec_max=dec_min+dec_step*j
            if i==njk-1:
                dec_max=dec_lim[-1]
                x2+=pixang[0]<dec_max
                n=sum(x1*x2)
                break
            x2+=pixang[0]<dec_max
            n=sum(x1*x2)
            if n>nexp_dec*1.005:
                print ('step change',j_step,n,nexp_dec,j,dec_max)
                j-=j_step
                j_step-=0.1
                n=0
                print ('step change',j_step,n,nexp_dec,j)
                if j_step<=0.05:
                    raise Exception('step problem')
                continue
            j+=j_step
            if dec_max>=dec_lim[-1]:
                print (i,dec_max,'dec lim reached')
                dec_max=dec_lim[-1]
                break
        dec_lim[i+1]=dec_max
        nsum_dec+=n
    #    nexp_ra=nexp_dec/njk
        nexp_ra=n/njk
        print (i,j,n)

        ra_lim=np.zeros(njk+1)
        ra_lim[0]=np.min(ra2)
        ra_lim[-1]=np.max(ra2)
        nsum_ra=0

        for k in np.arange(njk):
            ra_min=ra_lim[k]
            j=0
            n=0
            x3*=False
            x3+=pixang[1]>ra_min
            while n<nexp_ra*0.997:
                ra_max=ra_min+ra_step*j
                x4*=False
                if k==njk-1:
                    ra_max=ra_lim[-1]
                x4+=pixang[1]<ra_max
                n=sum(x1*x2*x3*x4)
                j+=1
                if ra_max>=ra_lim[-1]:
                    print (i,k,ra_max,'ra lim reached')
                    ra_max=ra_lim[-1]
                    break
                ra_lim[k+1]=ra_max
                jk[x1*x2*x3*x4]=ijk
                ijk+=1
                nsum_ra+=n

            ra_lim_all[i]=ra_lim
            nsum_ra_all[i]=nsum_ra
            print (i,j,nexp_dec,n,nsum_ra)

    mockmap2=np.zeros(12*nside**2)
    x=mockmap<-1000
    mockmap2[x]+=hp_masked
    x=mockmap>-1000
    mockmap2[x]=jk
    x=mockmap2!=-1
    mockmap2[x]=hp_masked
    jk_gal=mockmap2[mockpix]
    return jk_gal



def rotate_mask(mask=[],nside=512,frame_in='galactic',frame_out='',nest=False,nside_smooth_fact=4):
    order_out='Ring'
    if nest:
        order_out='nested'
    lm2=hp.pixelfunc.ud_grade(mask,nside_out=nside,pess=True,order_in='Ring',                                                                                        
                              order_out=order_out,dtype='int')
    mask2=lm2.astype('bool')
    if frame_out=='icrs':
        print('Assuming mask is in galactic. Converting to icrs')
        npix=np.arange(hp.nside2npix(nside))
        theta,phi=hp.pix2ang(nside=nside,ipix=npix)
        c=SkyCoord(b=(-theta+np.pi/2.)*units.radian, l=phi*units.radian,frame='galactic')  
        c_icrs=c.icrs
        gpix=hp.pixelfunc.ang2pix(nside=nside,nest=nest,theta=c_icrs.dec.radian*-1+np.pi/2.,                                                                         
                              phi=c_icrs.ra.radian)                                                                                                             
        lm3=np.zeros_like(lm2)
        lm3[gpix]=lm2
        lm4=hp.pixelfunc.ud_grade(np.float64(lm3),nside_out=nside/nside_smooth_fact,pess=False,order_in='Ring',                                                                                        
                              order_out=order_out)
        x=lm4>0
        lm4=x
        lm4=hp.pixelfunc.ud_grade(lm4,nside_out=nside,pess=True,order_in='Ring',                                                                                        
                              order_out=order_out,dtype='int')
        mask2=lm4.astype('bool')
    return mask2



def gal_jk2(mask=[],dat=[],nside=512,njk1=10,njk2=10,nest=False,frame_in='galactic',frame_use='galactic'):    
    mask2=rotate_mask(mask=mask,nside=nside,nest=nest,frame_in=frame_in,frame_out=frame_use)
    njk=njk1 #np.sqrt(njk_tot)                                                                                                                                       
    x=mask2>0                                                                                                                                                        
    nii=np.int64(sum(x)/njk+1)                                                                                                                                       
                                                                                                                                                                     
    jkmap=np.zeros_like(mask2,dtype='float64')-999                                                                                                                   
    ii=np.int64(np.arange(len(mask2)))                                                                                                                               
    ii=ii[x]                                                                                                                                                         
    for i in np.int64(np.arange(njk)):                                                                                                                               
        if i==nii%njk:                                                                                                                                               
            nii-=1                                                                                                                                                   
        indx=ii[i*nii:(i+1)*nii]                                                                                                                                     
        jkmap[indx]=i                                                                                                                                                
                                                                                                                                                                     
    #njk2=njk_tot/njk                                                                                                                                                
    jkmap2=np.zeros_like(jkmap)-999                                                                                                                                  
    ii=np.int64(np.arange(len(jkmap2)))                                                                                                                              
    for i in np.int64(np.arange(njk)):                                                                                                                               
        x=jkmap==i                                                                                                                                                   
        ii2=ii[x]                                                                                                                                                    
        add_f=0                                                                                                                                                      
        if i==0:#to get circle at galactic pole                                                                                                                      
            nii=np.int64(sum(x)/njk2)                                                                                                                                
            jkmap2[ii2[:nii]]=0                                                                                                                                      
            ii2=ii2[nii:]                                                                                                                                            
            add_f=1                                                                                                                                                  
        theta,phi=hp.pixelfunc.pix2ang(nside,ii2,nest=nest)                                                                                                          
        phi_bins=np.percentile(phi,np.linspace(0,100,njk2+1-add_f))                                                                                                  
        phi_bins[0]-=0.1                                                                                                                                             
        phi_bins[-1]+=0.1                                                                                                                                            
        for j in np.int64(np.arange(njk2-add_f)):                                                                                                                    
            xphi=phi>phi_bins[j]                                                                                                                                     
            xphi*=phi<phi_bins[j+1]                                                                                                                                  
            indx=ii2[xphi]                                                                                                                                           
            jkmap2[indx]=i*njk2+j+add_f                                               
    #dat_galactic=c.transform_to(Galactic)                                                                                                                           
    c=SkyCoord(ra=dat['RA']*units.degree, dec=dat['DEC']*units.degree,frame='icrs')   
    if frame_use=='icrs':
        gpix=hp.pixelfunc.ang2pix(nside=nside,nest=False,theta=c.dec.radian*-1+np.pi/2.,                                                                
                              phi=c.ra.radian)           
    elif frame_use=='galactic':
        dat_galactic=c.galactic                                                        
    jkgal=jkmap2[gpix] 
    return jkgal
