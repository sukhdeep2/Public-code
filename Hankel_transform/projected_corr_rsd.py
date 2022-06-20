#requires mcfit, pip install mcfit
# see https://arxiv.org/pdf/0911.4973.pdf... eq. 47-50
# for IA beta, see https://arxiv.org/pdf/1510.06752.pdf
import mcfit
from mcfit import P2xi
from scipy.interpolate import interp1d
from scipy.special import eval_legendre as legendre
from scipy import integrate
import numpy as np
from wigner_functions import *

class Projected_Corr_RSD():
    def __init__(self,rp=None,pi=None,pi_max=100,l=[0,2,4],k=None,spins=[0,2]):
        self.rp=rp
        self.pi=pi
        if rp is None:
            self.rp=np.logspace(-1,np.log10(200),100)
        if pi is None:
            self.pi=np.logspace(-3,np.log10(pi_max),250)
#            self.pi=np.append(0,self.pi)
        self.dpi=np.gradient(self.pi)
        self.piG,self.rpG=np.meshgrid(self.pi,self.rp)
        self.rG=np.sqrt(self.rpG**2+self.piG**2)
        self.muG=self.piG/self.rG
        self.L={}
        self.aL={}
        self.j={}
        for i in l:
            if 0 in spins:
                self.L[i]=legendre(i,self.muG)
            if 2 in spins:
                self.aL[i]=associated_legendre(i,2,self.muG)
            self.j[i]=P2xi(k,l=i)
        
    def alpha(self,l,beta1,beta2,spin=0):
        """
        For spin 2 case, beta2 must be beta_D. beta1 is set to -1
        """
        if spin==0:
            if l==0:
                return 1+1./3.*(beta1+beta2)+1./5*(beta1*beta2)
            elif l==2:
                return 2./3.*(beta1+beta2)+4./7.*(beta1*beta2)
            elif l==4:
                return 8./35.*(beta1*beta2)
        if spin==2:
            if l==2:
                return 1./3 + 1./21*beta2
            elif l==4:
                return 2./105*beta2


    def w_to_DS(self,rp=[],w=[]):
        DS0=2*w[0]*rp[0]**2
        return 2.*integrate.cumtrapz(w*rp,x=rp,initial=0)/rp**2-w+DS0/rp**2

    def get_xi_multipole(self,pk=[],l=[0,2,4]):
        xi={}
        r={}
        for i in l:
            r[i], xi[i] = self.j[i](pk)
        return r,xi
    
    def get_xi(self,pk=[],l=[0,2,4],spin=0):
        xi={}
        for i in l:
            ri, xi_i = self.j[i](pk)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            if spin==0:
                xi[i]=(xi_intp(self.rG)*self.L[i])@self.dpi
            elif spin==2:
                xi[i]=(xi_intp(self.rG)*self.aL[i])@self.dpi
            xi[i]*=2#one sided pi
        return xi

    def wgg_calc(self,f=0,bg=0,bg2=None,pk=[],xi=None,l=[0,2,4]):
        bg1=bg
        if bg2 is None:
            bg2=bg
        beta1=f/bg1
        beta2=f/bg2
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[0])
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg1*bg2).T
        return W

    def wgm_calc(self,f=0,bg=0,beta2=0,pk=[],xi=None,l=[0,2,4]):
        beta1=f/bg
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[0])
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg).T
        return W

    def DS_calc(self,f=0,bg=0,pk=[],xi=None,l=[2,4]): #same as wgp
        beta2=f/bg
        beta1=-1
        if xi is None:
            xi=self.get_xi(pk=pk,l=l,spin=2)
        DS=np.zeros_like(xi[0])
        for i in l:
            DS+=(xi[i].T*self.alpha(i,beta1,beta2,spin=2)*bg).T
        return DS
    
    def xi_rp_pi(self,f,bg1=0,bg2=None,pk=[],xi=None,l=[0,2,4],spin=0,rp=[],pi=[]):
        if bg2 is None:
            bg2=bg1
        beta1=f/bg1
        beta2=f/bg2
        if spin==2:
            beta1=-1
        r,xi_multipoles=self.get_xi_multipole(pk=pk,l=l)
        rp_g,pi_g=np.meshgrid(rp,pi)
        r_g=np.sqrt(rp_g**2+pi_g**2)
        mu_g=pi_g/r_g
        xi=np.zeros_like(r_g)
        for li in l:
            xi_li=np.interp(r_g,r[li],xi_multipoles[li],left=0,right=0)
            if spin==0:
                L_li=legendre(li,mu_g)*self.alpha(li,beta1,beta2,spin=0)
            elif spin==2:
                L_li=associated_legendre(li,2,mu_g)*self.alpha(li,beta1,beta2,spin=2)
            if li==2 and spin==2:
                xi_li*=-1
            xi+=xi_li*L_li
        return xi