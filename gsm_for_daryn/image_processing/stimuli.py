import numpy as np
from . import test_gratings as test
import image_processing as proc
import math

def make_OTUNE_filters(con,nfilt,nang,npha,freq,scale,tot,fdist):

    grats = np.array([[test.GRATC(c,a,freq,freq,tot + 2*fdist + 1) for a in np.linspace(0,2*np.pi,2)] for c in con])

    filt = np.array([[proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot) for g in C] for C in grats])

    filt = np.array([[proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in C] for C in filt])
    
    return filt

