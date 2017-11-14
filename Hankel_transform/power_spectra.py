import camb
import numpy as np
from camb import model, initialpower
#import cosmology
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c,G
from astropy import units as u

cosmo_h=cosmo.clone(H0=100)
c=c.to(u.km/u.second)

cosmo_fid=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'As':2.12e-09,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965})
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':5000}

class Power_Spectra():
    def __init__(self,cosmo_params=cosmo_fid,pk_params=pk_params,cosmo=cosmo,cosmo_h=None):
        self.cosmo_params=cosmo_params
        self.pk_params=pk_params
        self.cosmo=cosmo

        if not cosmo_h:
            self.cosmo_h=cosmo.clone(H0=100)
        else:
            self.cosmo_h=cosmo_h

    def Rho_crit(self,cosmo_h=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        rc=3*self.cosmo_h.H0**2/(8*np.pi*G2)
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc

    def sigma_crit(self,zl=[],zs=[],cosmo_h=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        ds=cosmo_h.comoving_transverse_distance(zs)
        dl=cosmo_h.comoving_transverse_distance(zl)
        ddls=1-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        w=(3./2.)*((cosmo_h.H0/c)**2)*(1+zl)*dl/self.Rho_crit(cosmo_h)
        sigma_c=1./(ddls*w)
        x=ddls<=0 #zs<zl
        sigma_c[x]=np.inf
        return sigma_c

    def camb_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        #Set up a new set of parameters for CAMB
        if not cosmo_params:
            cosmo_params=self.cosmo_params
        if not pk_params:
            pk_params=self.pk_params

        pars = camb.CAMBparams()
        h=cosmo_params['h']

        pars.set_cosmology(H0=h*100,
                            ombh2=cosmo_params['Omb']*h**2,
                            omch2=(cosmo_params['Om']-cosmo_params['Omb'])*h**2,
                            mnu=cosmo_params['mnu'],tau=cosmo_params['tau']
                            ) #    omk=cosmo_params['Omk'], )

        #stdout=np.copy(sys.stdout)
        #sys.stdout = open(os.devnull, 'w')

        pars.InitPower.set_params(ns=cosmo_params['ns'], r=0,As =cosmo_params['As']) #
        if return_s8:
            z_t=np.sort(np.unique(np.append([0],z).flatten()))
        else:
            z_t=np.array(z)
        pars.set_matter_power(redshifts=z_t,kmax=pk_params['kmax'])

        #sys.stdout = sys.__stdout__
        #sys.stdout=stdout

        if pk_params['non_linear']==1:
            pars.NonLinear = model.NonLinear_both
        else:
            pars.NonLinear = model.NonLinear_none

        results = camb.get_results(pars) #This is the time consuming part.. pk add little more (~5%).. others are negligible.

        kh, z2, pk =results.get_matter_power_spectrum(minkh=pk_params['kmin'],
                                                        maxkh=pk_params['kmax'],
                                                        npoints =pk_params['nk'])
        if not np.all(z2==z_t):
            raise Exception('CAMB changed z order',z2,z_mocks)

        if return_s8:
            s8=results.get_sigma8()
            if len(s8)>len(z):
                return pk[1:],kh,s8[-1]
            else:
                return pk,kh,s8[-1]
        else:
            return pk,kh

    def cl_z(self,z=[],l=np.arange(2000)+1,pk_params=None,cosmo_h=None,cosmo=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        nz=len(z)
        nl=len(l)

        i=0
        pk=np.array([])
        z_step=140 #camb cannot handle more than 150 redshifts
        while pk.shape[0]<nz:
            pki,kh=self.camb_pk(z=z[i:i+z_step],pk_params=pk_params)
            pk=np.vstack((pk,pki)) if pk.size else pki
            i+=z_step

        cls=np.zeros((nz,nl),dtype='float32')*u.Mpc**2
        for i in np.arange(nz):
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value
            pk_int=interp1d(lz,pk[i]/DC_i**2,bounds_error=False,fill_value=0)
            cls[i][:]+=pk_int(l)*u.Mpc*(c/(cosmo_h.efunc(z[i])*cosmo_h.H0))
        return cls

    def kappa_cl(self,zl_min=0,zl_max=1100,n_zl=10,log_zl=False,
                zs1=[1100],p_zs1=[1],zs2=[1100],p_zs2=[1],
                pk_params=None,cosmo_h=None,l=np.arange(2001)):
        if not cosmo_h:
            cosmo_h=self.cosmo_h

        if log_zl:
            zl=np.logspace(np.log10(max(zl_min,1.e-2)),np.log10(zl_max),n_zl)
        else:
            zl=np.linspace(zl_min,zl_max,n_zl)

        clz=self.cl_z(z=zl,l=l,cosmo_h=cosmo_h,pk_params=pk_params)

        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
        sigma_c1=rho/self.sigma_crit(zl=zl,zs=zs1,cosmo_h=cosmo_h)
        sigma_c2=rho/self.sigma_crit(zl=zl,zs=zs2,cosmo_h=cosmo_h)

        dzl=np.gradient(zl)
        dzs1=np.gradient(zs1) if len(zs1)>1 else 1
        dzs2=np.gradient(zs2) if len(zs2)>1 else 1

        cl_zs_12=np.einsum('ji,ki,il',sigma_c2,sigma_c1*dzl,clz)#integrate over zl..
        cl=np.dot(p_zs2*dzs2,np.dot(p_zs1*dzs1,cl_zs_12))
        cl/=np.sum(p_zs2*dzs2)*np.sum(p_zs1*dzs1)
        f=(l+0.5)**2/(l*(l+1.)) #correction from Kilbinger+ 2017
        cl*=f
        #cl*=2./np.pi #comparison with CAMB requires this.
        return l,cl
