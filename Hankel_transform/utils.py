import numpy as np
from scipy.optimize import curve_fit,minimize
from scipy.stats import binned_statistic as BS
from multiprocessing import Pool,Process


def check_empty(arr): # True is empty
    try:
        #print not arr.any()
        try:
            return not arr.any()
        except:
            if arr.size!=0: #for rec array
                return False
            else:
                return True
    except:
        #print not bool(arr)
        return not bool(arr) #for other python types, including dictionary

def curve_fit2(func,dat,p0,sigma,*params):
    dat=np.nan_to_num(dat)
    sigma=np.nan_to_num(sigma)
    p,pe=curve_fit(func,(params[0]),dat,p0=p0,sigma=sigma)
#    print p
    return p

def minimize2(func,p0,*args):
    pp=minimize(func,p0,args=args,method='Powell',tol=1.e-6)
    return pp.x

def calc_stat_quick(xvar='',yvar='',wt_var='',dat=[],datx=[],daty=[],wt=[],N=20,x_min=0,x_max=0,do_wt=0,rms=0,xlog=0,stat='mean',**kwargs):
    bin_func=np.linspace
    if datx==[]:
        datx=dat[xvar]
    if daty==[]:
        daty=dat[yvar]

    if xlog!=0:
        bin_func=np.logspace
    if x_min==0 and x_max==0:
        x_min=min(datx)
        x_max=max(datx)
    if xlog==0:
        zz=np.linspace(x_min,x_max,N+1)
    else:
        zz=np.logspace(np.log10(x_min),np.log10(x_max),N+1)   

    m,zb,bi=BS(datx,(daty**(rms+1))/(rms+1),bins=zz,statistic=stat)
    zb=0.5*(zb[1:]+zb[:-1])
    return zb,m,0


def calc_mean_quick(xvar='',yvar='',wt_var='',dat=[],datx=[],daty=[],wt=[],N=20,x_min=0,x_max=0,do_wt=0,rms=0,xlog=0,**kwargs):
    bin_func=np.linspace
    if datx==[]:
        datx=dat[xvar]
    if daty==[]:
        daty=dat[yvar]
    if xlog!=0:
        bin_func=np.logspace
    if x_min==0 and x_max==0:
        x_min=min(datx)
        x_max=max(datx)
    if xlog==0:
        zz=np.linspace(x_min,x_max,N+1)
    else:
        zz=np.logspace(np.log10(x_min),np.log10(x_max),N+1)
    zc,zb=np.histogram(datx,bins=zz)
    zc=np.float64(zc)
    zc2,zb2=np.histogram(datx,weights=(daty**(rms+1.))/(rms+1),bins=zz)
    mm=zc2/zc
    mm=mm**(1./(rms+1.))
    #zc2=zc2**(1./(rms+1.))
    #zc=zc**(1./(rms+1.))
    zb=0.5*(zb[1:]+zb[:-1])
    return zb,mm,0

def calc_mean(xvar='',yvar='',wt_var='',dat=[],datx=[],daty=[],wt=[],N=20,x_min=0,x_max=0,do_wt=0,rms=0,xlog=0,**kwargs):
    if datx==[]:
        datx=dat[xvar]
    if daty==[]:
        daty=dat[yvar]
    if do_wt!=0 and wt==[]:
        wt=dat[wt_var]
    if x_min==0 and x_max==0:
        zz=np.linspace(min(datx),max(datx),N+1)
    else:
        zz=np.linspace(x_min,x_max,N+1)
    if xlog!=0:
        zz=np.logspace(np.log10(x_min),np.log10(x_max),N+1)
    if do_wt==0:
        mean_func=np.mean
    if do_wt!=0:
        mean_func=wt_mean_var
    zz2=[]
    m=[]
    sig=[]
    for i in np.arange(N):
        x2=datx<zz[i+1]
        x3=datx>zz[i]
        x=x2*x3
        #print i,sum(x)
        if sum(x)==0:
            m=np.append(m,0)
            sig=np.append(sig,0)
        if do_wt==0 and sum(x)>0:
            m=np.append(m,np.mean(daty[x]**(rms+1)))
            sig=np.append(sig,np.sqrt(np.var(daty[x]**(rms+1))/sum(x)))
        if do_wt!=0 and sum(x)>0:
            mt,vt,st=wt_mean_var(dat=daty[x],wt=wt[x])
            m=np.append(m,mt)
            sig=np.append(sig,st)
        if xlog==0:
            zz2=np.append(zz2,zz[i]+zz[i+1])
        else:
            zz2=np.append(zz2,zz[i]*zz[i+1])
    if xlog==0:
        zz2/=2.0
    else:
        zz2=np.sqrt(zz2)
    if rms!=0:
        m=np.sqrt(m)
    return zz2,m,sig

def wt_mean_var(dat=[],wt=[]):
    mean=np.sum(dat*wt)/np.sum(wt)
    var=np.sum(wt*(dat-mean)**2)/(np.sum(wt)-np.sum(wt**2)/np.sum(wt))
    return mean,var,np.sqrt(var/len(dat))

def wt_median(dat=[],wt=[]):
    e=array(dat,dtype=[('e','float64')])
    w=array(wt,dtype=[('w','float64')])
    xx=merge_arrays((e,w),flatten=True)
    xx2=np.sort(xx,order='e')
    y=np.sum(wt)
    y2=0
    for i in np.arange(len(LP_shape['6wt'])):
        y2+=xx2[i][1]
        if y2>=y/2.0:
            #print i,xx2[i]
            break
#    print i,y,y2,xx2[i],len(LP_shape['6wt'])
    return i,xx2[i]['e']

def calc_mean_jki(jkr=None,**kwargs):
    x=dat_jk!=i
    if do_wt!=0:
        wt_temp=wt[x]
    #zbin,means,sig=
    return mean_func(xvar=xvar,yvar=yvar,datx=datx[x],daty=daty[x],wt=wt_temp,N=N,x_min=x_min,x_max=x_max,do_wt=do_wt,**kwargs)
        
def calc_mean_jk(mean_func=calc_mean_quick,xvar='',yvar='',jk='jk',wt_var='',dat=[],datx=[],daty=[],wt=[],dat_jk=[],x_min=0,x_max=0,N=10,njk=100,do_wt=0,ncpu=1
                 ,**kwargs):
    if datx==[]:
        datx=dat[xvar]
    if daty==[]:
        daty=dat[yvar]
    if dat_jk==[]:
        dat_jk=dat[jk]
    if wt==[] and do_wt!=0:
        wt=dat[wt_var]
    jkr=np.arange(njk) #np.unique(dat_jk)
    means={}
    zbin={}
    mean=[]
    wt_temp=[]
    j=0

    local_kwargs=copy.deepcopy(locals())
    print(list(local_kwargs.keys()),list(kwargs.keys()))
    ff=partial(calc_mean_jki,**local_kwargs)
    pool=Pool(ncpu)
    out_jki=pool.map(ff,jkr)
    
    zbin['dat'],means['dat'],sig=mean_func(xvar=xvar,yvar=yvar,datx=datx,daty=daty,wt=wt,N=N,x_min=x_min,x_max=x_max,do_wt=do_wt,**kwargs)
    for i in jkr:
        x=dat_jk!=i
        if do_wt!=0:
            wt_temp=wt[x]
        zbin[i],means[i],sig=mean_func(xvar=xvar,yvar=yvar,datx=datx[x],daty=daty[x],wt=wt_temp,N=N,x_min=x_min,x_max=x_max,do_wt=do_wt,**kwargs)
        if j==0:
            mean=means[i]
            j+=1
            continue
        mean=np.column_stack((mean,means[i]))
    means['jk']=np.mean(mean,axis=1)
    means['var']=np.var(mean,axis=1)
    means['jk_err']=np.sqrt(np.var(mean,axis=1)*99.*99./100.)
    return zbin[0],means


def number_density(z=[],N=200,Area=8000.,wt=[],cosmology=[],**kwargs):
    if wt==[]:
        wt=np.ones_like(z)

    zc,zb=np.histogram(z,bins=N,weights=wt)
    zb_mean=0.5*(zb[1:]+zb[:-1])
    vol=cosmology.comoving_volume(z=zb)
    nc=np.zeros_like(zc,dtype='float64')
    d2r=np.pi/180.
    Area*=d2r**2
    zc=np.array(zc,dtype='float64')
    for i in np.arange(N):
        nc[i]=float(zc[i])/(Area*vol[i])
    return zb_mean,nc

def number_density_jk(data=[],N=200,Area=8000.,wt=[],cosmology=[],**kwargs):
    nc={}
    Area*=99./100.
    for i in np.arange(100):
        x=data['jk']!=i
        zb,nc[i]=number_density(z=data['z'][x],N=N,Area=Area,wt=wt[x],cosmology=cosmology,**kwargs)
    nc=jk_mean(p=nc)
    return zb,nc
