#requires mcfit, pip install mcfit
# see https://arxiv.org/pdf/0911.4973.pdf... eq. 47-50
# for IA beta, see https://arxiv.org/pdf/1510.06752.pdf
import mcfit
from mcfit import P2xi
from scipy.interpolate import interp1d
from scipy.special import eval_legendre as legendre
from scipy import integrate
import numpy as np

class Projected_Corr_RSD():
    def __init__(self,rp=None,pi=None,pi_max=100,l=[0,2,4],k=None):
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
        self.j={}
        for i in l:
            self.L[i]=legendre(i,self.muG)
            self.j[i]=P2xi(k,l=i)
        
    def alpha(self,l,beta1,beta2):
        if l==0:
            return 1+1./3.*(beta1+beta2)+1./5*(beta1*beta2)
        elif l==2:
            return 2./3.*(beta1+beta2)+4./7.*(beta1*beta2)
        elif l==4:
            return 8./35.*(beta1*beta2)

    def w_to_DS(self,rp=[],w=[]):
        DS0=2*w[0]*rp[0]**2
        return 2.*integrate.cumtrapz(w*rp,x=rp,initial=0)/rp**2-w+DS0/rp**2

    def get_xi(self,pk=[],l=[0,2,4]):
        xi={}
        for i in l:
            ri, xi_i = self.j[i](pk)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            xi[i]=(xi_intp(self.rG)*self.L[i])@self.dpi
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

    def wgm_calc(self,f=0,bg=0,beta2=0,pk=[],xi=None,l=[0,2,4],do_DS=False):
        beta1=f/bg
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[0])
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg).T
        if do_DS:
            W=self.w_to_DS(rp=self.rp,w=W)
        return W
