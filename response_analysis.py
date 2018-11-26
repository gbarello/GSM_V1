import numpy as np
import sys
import json

import simtools as sim

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

ext = ".jpg"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 25
plt.rcParams['lines.linewidth'] = 4.

color1 = (1,1,1)
color2 = (.5,.5,.5)
color3 = (.25,.25,.25)

LEG = False
COSplot = False
WTAplot = False

def rec(x):
    return (x + np.abs(x))/2

def resp(f):
    ll = len(f)/2
    return np.sqrt(rec(f[::2])**2 + rec(f[1::2])**2)

def MGSM_resp(data):

    with open(data + "model_params.json", 'r') as fp:
        model_params = json.load(fp)

    print(model_params)
    
    ntrial = model_params["ntrial"]
    n_cos_a = model_params["n_cos_a"]

    con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])

    print(ntrial)
    
    size_tuning_noisy = np.array([resp(x) for x in np.loadtxt(data + "size_tuning_noisy.csv")])
    size_tuning_clean = np.array([resp(x) for x in np.loadtxt(data + "size_tuning_clean.csv")])

    size_tuning_SP_noisy = np.loadtxt(data + "size_tuning_SP_noisy.csv")
    size_tuning_SP_clean = np.loadtxt(data + "size_tuning_SP_clean.csv")
    
    print(size_tuning_SP_noisy.shape)
    print(size_tuning_SP_clean.shape)

    size_tuning_noisy = np.reshape(size_tuning_noisy,[5,model_params["ntrial"],len(con),size_tuning_noisy.shape[0]/(5*len(con)*ntrial),4])
    size_tuning_clean = np.reshape(size_tuning_clean,[5,1,len(con),size_tuning_clean.shape[0]/(5*len(con)),4])

    print(size_tuning_SP_noisy.shape)

    size_tuning_SP_noisy = np.reshape(size_tuning_SP_noisy,[5,model_params["ntrial"],len(con),size_tuning_SP_noisy.shape[0]/(5*len(con)*ntrial),5]).mean(axis = 1)
    size_tuning_SP_clean = np.reshape(size_tuning_SP_clean,[5,1,len(con),size_tuning_SP_clean.shape[0]/(5*len(con)),5]).mean(axis = 1)
    
    ST_n_mean = size_tuning_noisy.mean(axis = 1)
    ST_n_SD = size_tuning_noisy.std(axis = 1)
    
    ST_c_mean = size_tuning_clean.mean(axis = 1)
    ST_c_SD = size_tuning_clean.std(axis = 1)
    #import FF response function
    
    full_field_noisy = np.array([resp(x) for x in np.loadtxt(data + "full_field_noisy.csv")])
    full_field_clean = np.array([resp(x) for x in np.loadtxt(data + "full_field_clean.csv")])
    
    full_field_noisy = np.reshape(full_field_noisy,[5,model_params["ntrial"],len(con),4])
    full_field_clean = np.reshape(full_field_clean,[5,1,len(con),4])
    
    FF_n_mean = full_field_noisy.mean(axis = 1)
    FF_n_SD = full_field_noisy.std(axis = 1)
    
    FF_c_mean = full_field_clean.mean(axis = 1)
    FF_c_SD = full_field_clean.std(axis = 1)

    for k in range(len(FF_n_mean[2])):
        print("{}\t{}".format(FF_n_mean[-2,k,0],FF_c_mean[-2,k,0]))

#import the surround surppression data
    
    surr_supp_noisy = np.array([resp(x) for x in np.loadtxt(data + "surr_supp_noisy.csv")])
    surr_supp_clean = np.array([resp(x) for x in np.loadtxt(data + "surr_supp_clean.csv")])
    
    surr_supp_SP_noisy = np.loadtxt(data + "surr_supp_SP_noisy.csv")
    surr_supp_SP_clean = np.loadtxt(data + "surr_supp_SP_clean.csv")
    
    surr_supp_noisy = np.reshape(surr_supp_noisy,[5,model_params["ntrial"],5,len(con),2,4])
    surr_supp_clean = np.reshape(surr_supp_clean,[5,1,5,len(con),2,4])

#    surr_supp_SP_noisy = np.reshape(surr_supp_SP_noisy,[5,model_params["ntrial"],5,len(con),2,5]).mean(axis = 1)
#    surr_supp_SP_clean = np.reshape(surr_supp_SP_clean,[5,1,5,len(con),2,5]).mean(axis = 1)
    
    SS_n_mean = surr_supp_noisy.mean(axis = 1)
    SS_n_SD = surr_supp_noisy.std(axis = 1)
    
    SS_c_mean = surr_supp_clean.mean(axis = 1)
    SS_c_SD = surr_supp_clean.std(axis = 1)
    
    #COS data
    if COSplot:
        FF_COS_noisy = np.array([resp(x) for x in np.loadtxt(data + "full_field_COS_noisy.csv")])
        FF_COS_clean = np.array([resp(x) for x in np.loadtxt(data + "full_field_COS_clean.csv")])
        
        FF_COS_noisy = np.reshape(FF_COS_noisy,[5,model_params["ntrial"],n_cos_a,len(con),len(con),4])
        FF_COS_clean = np.reshape(FF_COS_clean,[5,1,n_cos_a,len(con),len(con),4])
        
        FFC_n_mean = FF_COS_noisy.mean(axis = 1)
        FFC_n_SD = FF_COS_noisy.std(axis = 1)
        
        FFC_c_mean = FF_COS_clean.mean(axis = 1)
        FFC_c_SD = FF_COS_clean.std(axis = 1)
        
        FFAI_n = np.array([[FFC_n_mean[:,:,k,j]/(FFC_n_mean[:,:,k,0] + FFC_n_mean[:,:,0,j]) for j in range(len(con))] for k in range(len(con))])
        FFAI_c = np.array([[FFC_c_mean[:,:,k,j]/(FFC_c_mean[:,:,k,0] + FFC_c_mean[:,:,0,j]) for j in range(len(con))] for k in range(len(con))])
        
        FFAI_n = np.transpose(FFAI_n,[2,3,0,1,4])
        FFAI_c = np.transpose(FFAI_c,[2,3,0,1,4])


    if WTAplot:
        FF_WTA_noisy = np.array([resp(x) for x in np.loadtxt(data + "full_field_WTA_noisy.csv")])
        FF_WTA_clean = np.array([resp(x) for x in np.loadtxt(data + "full_field_WTA_clean.csv")])
        
        FF_WTA_noisy = np.reshape(FF_WTA_noisy,[5,model_params["ntrial"],len(con),32,2,4])
        FF_WTA_clean = np.reshape(FF_WTA_clean,[5,1,len(con),32,2,4])
        
        FFW_n_mean = FF_WTA_noisy.mean(axis = 1)
        FFW_n_SD = FF_WTA_noisy.std(axis = 1)
        
        FFW_c_mean = FF_WTA_clean.mean(axis = 1)
        FFW_c_SD = FF_WTA_clean.std(axis = 1)
    
    #FF Shape: [k,con,condition,f_ang]
    print("done importing")

    cl = np.argmin(np.abs(con - .01))
    
    fn = 0
#make FF figures
    
    print(len(con))
    
    KK = [.5,.75,1,1.5,2]
    
    for k in range(len(FF_n_mean)):
        fn += 1
        plt.figure(fn)
        
        l1 = plt.plot(con[cl:],FF_n_mean[k,cl:,0],"k",label = "noisy")
        l2 = plt.plot(con[cl:],FF_c_mean[k,cl:,0],"k--",label = "noiseless")

        if LEG:
            plt.legend(loc = 2)
        
        plt.xlabel("Contrast")
        plt.ylabel("Response (a.u.)")
        plt.xscale("log")
        plt.savefig(data+"FF_plot_{}".format(k)+ext)

#plot of all FF responses
        fn += 1
        plt.figure(fn)

        l = []

        for k in range(len(FF_n_mean)):
            col = (.5 + .5*(float(k)/(len(FF_n_mean)-1)))
            plt.plot(con[cl:],FF_n_mean[k,cl:,0],color = (col,col,col),label = "k = {}".format(KK[k]))

        if LEG:
            plt.legend(loc = 2)
            
        plt.xlabel("Contrast")
        plt.ylabel("Response (a.u.)")
        plt.xscale("log")

        plt.tight_layout()

        plt.savefig(data+"FF_plot_{}".format(len(FF_n_mean))+ext)

#make SS figures
#SS Shape: [k,d_ang,con,condition,f_ang]

    for k in range(len(SS_n_mean)):
        for d in range(len(SS_n_mean[k])):
            fn += 1
            plt.figure(fn)
            l1 = plt.plot(con[cl:],SS_n_mean[k,d,cl:,0,0],"r",label = "surr. = 0")
            l2 = plt.plot(con[cl:],SS_n_mean[k,d,cl:,1,0],"k",label = "surr. = .5")

            if LEG:
                plt.legend(loc = 2)

            plt.plot(con[cl:],SS_c_mean[k,d,cl:,0,0],"r--")
            plt.plot(con[cl:],SS_c_mean[k,d,cl:,1,0],"k--")
            
            plt.xlabel("Contrast")
            plt.ylabel("Response (a.u.)")
            plt.xscale("log")

            plt.tight_layout()

            plt.savefig(data+"SS_plot_{}_{}".format(k,d)+ext)

#            fn += 1
#            plt.figure(fn)
#            l1 = plt.plot(con[cl:],1. - surr_supp_SP_noisy[k,d,cl:,0,0],"r",label = "surr. = 0")
#            l2 = plt.plot(con[cl:],1. - surr_supp_SP_clean[k,d,cl:,0,0],"r--",label = "surr. = .5")

#            l1 = plt.plot(con[cl:],1. - surr_supp_SP_noisy[k,d,cl:,1,0],"k",label = "surr. = 0")
#            l2 = plt.plot(con[cl:],1. - surr_supp_SP_clean[k,d,cl:,1,0],"k--",label = "surr. = .5")

#            plt.xlabel("Contrast")
#            plt.ylabel("Response (a.u.)")
#            plt.xscale("log")
#            plt.yscale("log")

#            plt.tight_layout()

#            plt.savefig(data+"SS_SP_plot_{}_{}".format(k,d)+ext)
                
            #ST Shape: [k,con,condition,f_ang]

    SS1 = np.argmin(np.abs(con - .05))
    SS2 = np.argmin(np.abs(con - .25))
    SS3 = -1

    for k in range(len(ST_n_mean)):
        fn += 1
        plt.figure(fn)
        for g in ST_n_mean[k]:
            plt.plot(np.arange(len(ST_n_mean[k,0])),g[:,0])

        plt.savefig(data+"all_ST_plot_{}".format(k)+ext)        

        fn += 1
        plt.figure(fn)
        plottab = []
        nnplottab = []
        ind = np.argmin(np.abs(con - .01))

        for g in range(ind,len(ST_n_mean[k])):
            if np.max(ST_n_mean[k,g,:,0]) > 0:
                plottab.append([con[g],np.argmax(ST_n_mean[k,g,:,0])])
            nnplottab.append([con[g],np.argmax(ST_c_mean[k,g,:,0])])

        plottab = np.array(plottab)
        nnplottab = np.array(nnplottab)

        plt.plot(plottab[:,0],plottab[:,1],'r')
        plt.plot(nnplottab[:,0],nnplottab[:,1],'r--')

        plt.xlabel("contrast")
        plt.ylabel("RF size (px.)")

        plt.xscale("log")
        plt.tight_layout()

        plt.savefig(data+"RFsize_plot_{}".format(k)+ext)

        fn += 1
        plt.figure(fn)
        
        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),ST_n_mean[k,SS1,:,0],"r",label = "con. = {}".format(con[7]))
        l2 = plt.plot(np.arange(len(ST_n_mean[k,0])),ST_n_mean[k,SS2,:,0],"g",label = "con. = {}".format(con[10]))
        l3 = plt.plot(np.arange(len(ST_n_mean[k,0])),ST_n_mean[k,SS3,:,0],"b",label = "con. = {}".format(con[15]))

        if LEG:
            plt.legend(loc = 1)

        plt.plot(np.arange(len(ST_c_mean[k,0])),ST_c_mean[k,SS1,:,0],"r--")
        plt.plot(np.arange(len(ST_c_mean[k,0])),ST_c_mean[k,SS2,:,0],"g--")
        plt.plot(np.arange(len(ST_c_mean[k,0])),ST_c_mean[k,SS3,:,0],"b--")
        
        plt.xlabel("Size")
        plt.ylabel("Response (a.u.)")

        plt.tight_layout()

#        plt.set_xrange([0,len(ST_n_mean[k,0])])
#        plt.set_xrange([0,1.5*np.max(ST_n_mean[k])])

        plt.savefig(data+"ST_plot_{}".format(k)+ext)

        fn += 1
        plt.figure(fn)

        pnts = []
        for x in ST_n_mean[k]:
            pnts.append(1. - (x[-1,0]/np.max(x[:,0])))

        nnpnts = []
        for x in ST_c_mean[k]:
            nnpnts.append(1. - (x[-1,0]/np.max(x[:,0])))
    
        plt.plot(con,pnts,'r')
        plt.plot(con,nnpnts,'r--')

        plt.xlabel("Contrast")
        plt.ylabel("Suppression Index")

        plt.tight_layout()

        plt.savefig(data+"ST_SI_{}".format(k)+ext)

        fn += 1
        plt.figure(fn)

        ST_n_mean[k,len(con)/4,:,0]

        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_noisy[k,SS1,:,0],"r",label = "surr. = 0")
        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_noisy[k,SS2,:,0],"g",label = "surr. = 0")
        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_noisy[k,SS3,:,0],"b",label = "surr. = 0")

        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_clean[k,SS1,:,0],"r--",label = "surr. = 0")
        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_clean[k,SS2,:,0],"g--",label = "surr. = 0")
        l1 = plt.plot(np.arange(len(ST_n_mean[k,0])),1. - size_tuning_SP_clean[k,SS3,:,0],"b--",label = "surr. = 0")
        
        plt.xlabel("Contrast")
        plt.ylabel("Response (a.u.)")
        plt.xscale("log")
        plt.yscale("log")
        
        plt.tight_layout()
        
        plt.savefig(data+"ST_SP_plot_{}".format(k)+ext)


    if COSplot:
#COS shape: [k,d_ang,tar_con,mas_con,f_ang]
        for k in range(len(FFC_n_mean)):
            for d in range(len(FFC_n_mean[k])):
                fn += 1
                plt.figure(fn)
                
                l1 = plt.plot(con[cl:],[FFAI_n[k,d,i,i,0] for i in range(cl,len(FFAI_n[k,d]))],"r",label = "noisy")
                
                l2 = plt.plot(con[cl:],[FFAI_c[k,d,i,i,0] for i in range(cl,len(FFAI_c[k,d]))],"r--",label = "noiseless")
                
                plt.plot([0] + con,[1. for i in range(len([0] + con))],"k--")
                
                if LEG:
                    plt.legend()
                    
                plt.xlabel("Contrast")
                plt.ylabel("Response (a.u.)")
                plt.xscale("log")

                plt.tight_layout()
                
                plt.savefig(data+"COS_plot_{}_{}".format(k,d)+ext)
            
        for k in range(len(FFC_n_mean)):
            for d in range(len(FFC_n_mean[k])):
                fn += 1
                plt.figure(fn)
                
                l1 = plt.plot(con[cl:],[FFC_n_mean[k,d,i,0,0] for i in range(cl,len(con))],"r",label = "mask = 0")
                
                l2 = plt.plot(con[cl:],[FFC_n_mean[k,d,i,-1,0] for i in range(cl,len(con))],"k",label = "mask = {}".format(con[-1]))
                
                if LEG:
                    plt.legend(loc = 2)
                    
                plt.plot(con[cl:],[FFC_c_mean[k,d,i,0,0] for i in range(cl,len(con))],"r--")
            
                plt.plot(con[cl:],[FFC_c_mean[k,d,i,27,0] for i in range(cl,len(con))],"k--")
            
                plt.xlabel("Contrast")
                plt.ylabel("Response (a.u.)")
                plt.xscale("log")
                
                plt.tight_layout()
                
                plt.savefig(data+"COS_CRF_{}_{}".format(k,d)+ext)
                
        ind= np.array([-1,3*len(con)/4,len(con)/2,len(con)/4])
        
        if WTAplot:
            for k in range(len(FFW_n_mean)):
                
                fn += 1
                plt.figure(fn)
                
                #            plt.xlabel("Target Contrast")
                #            plt.ylabel("Mask Contrast")
                RL = int(len(FFW_c_mean[k,0])/4)
                
                for c1 in range(len(ind)):
                    
                    plt.subplot(2,2,c1 + 1)
                    #[k,con,angle,cond,ori]
                    
                
                    
                    #                plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
                    #                         (
                    #                         np.roll(np.append(FFW_n_mean[k,ind[c1],:,0,0],FFW_n_mean[k,ind[c1],-1,0,0]),RL)
                    #                         +
                    #                         np.roll(np.append(FFW_n_mean[k,0,:,1,0],FFW_n_mean[k,0,-1,1,0]),RL)
                    #                         )
                    #                         ,"k--")
                    
                    
                    plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),np.roll(np.append(FFW_n_mean[k,0,:,1,0],FFW_n_mean[k,0,-1,1,0]),RL),"g")
                    plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),np.roll(np.append(FFW_n_mean[k,ind[c1],:,0,0],FFW_n_mean[k,ind[c1],-1,0,0]),RL),"b")
                    
                    plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),np.roll(np.append(FFW_n_mean[k,ind[c1],:,1,0],FFW_n_mean[k,ind[c1],-1,1,0]),RL),"k")
                
                
                    plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        left='off',      # ticks along the bottom edge are off
                        right='off',         # ticks along the top edge are off
                        labelbottom='off',
                        labeltop='off',
                        labelleft='off',
                        labelright='off') # labels along the bottom edge are off
                    
                    plt.savefig(data+"COS_WTA_{}".format(k)+ext)
        
        for k in range(len(FFC_n_mean)):
            for d in range(len(FFC_n_mean[k])):
                fn += 1
                plt.figure(fn)
                
                temp = np.array((FFC_n_mean[k,d,:,:,0] + np.transpose(FFC_n_mean[k,d,:,:,0]))/2)
                plt.pcolor(np.array(con),np.array(con),temp,label = "A.I.")
                
                plt.colorbar()
                
                plt.contour(np.array(con),np.array(con),temp,label = "A.I.",colors = 'k',linewidths = 1)
                
                #            plt.xscale('log')
                #            plt.yscale('log')
                
                plt.xlabel("Target Contrast")
                plt.ylabel("Mask Contrast")
                
                plt.tight_layout()
                plt.axes().set_aspect('equal', 'datalim')
                
                plt.savefig(data+"COS_CON_{}_{}".format(k,d)+ext)

                fn += 1
                plt.figure(fn)
                
                temp = np.array((FFC_c_mean[k,d,:,:,0] + np.transpose(FFC_c_mean[k,d,:,:,0]))/2)
                plt.pcolor(np.array(con),np.array(con),temp,label = "A.I.")
                
                plt.colorbar()
                
                plt.contour(np.array(con),np.array(con),temp,label = "A.I.",colors = 'k',linewidths = 1)
                
                #            plt.xscale('log')
                #            plt.yscale('log')
                
                plt.xlabel("Target Contrast")
                plt.ylabel("Mask Contrast")
                
                plt.tight_layout()
                plt.axes().set_aspect('equal', 'datalim')
                
                plt.savefig(data+"nnCOS_CON_{}_{}".format(k,d)+ext)
    
#I still need to add: 

#D) max(AI) and min(AI) vs. k (avg SNR) and w (what was w?? Is it related to the spatial frequency of filters, or say their size??)
#E) contrast of tansition from super to sub-additive vs. k and w

def GSM_resp(data):
#COS data

    plt.figure(1)

    fn = 1

    with open(data + "model_params.json", 'r') as fp:
        model_params = json.load(fp)

    ntrial = model_params["ntrial"]
    n_cos_a = model_params["n_cos_a"]
    print(ntrial)
    print(n_cos_a)
    print(model_params["nang"])
    
    con = np.array([.005 * x for x in range(20)] + [.1 + .05*x for x in range(19)])

    ind = np.arange(len(con))
    
    FF_COS_noisy = np.array([resp(x) for x in np.loadtxt(data + "full_field_COS_noisy.csv")])
    FF_COS_clean = np.array([resp(x) for x in np.loadtxt(data + "full_field_COS_clean.csv")])

    print(FF_COS_noisy.shape[0]/(5*n_cos_a*7*7))
    
    FF_COS_noisy = np.reshape(FF_COS_noisy,[5,model_params["ntrial"],n_cos_a,len(ind),len(ind),3,model_params["nang"]])
    FF_COS_clean = np.reshape(FF_COS_clean,[5,1,n_cos_a,len(ind),len(ind),3,model_params["nang"]])
    
    FFC_n_mean = FF_COS_noisy.mean(axis = 1)
    FFC_n_SD = FF_COS_noisy.std(axis = 1)
    
    FFC_c_mean = FF_COS_clean.mean(axis = 1)
    FFC_c_SD = FF_COS_clean.std(axis = 1)
    
    FFAI_n = FFC_n_mean[:,:,:,:,2]/(FFC_n_mean[:,:,:,:,1] + FFC_n_mean[:,:,:,:,0])
    FFAI_c = FFC_c_mean[:,:,:,:,2]/(FFC_c_mean[:,:,:,:,1] + FFC_c_mean[:,:,:,:,0])

    FF_WTA_noisy = np.array([resp(x) for x in np.loadtxt(data + "full_field_WTA_noisy.csv")])
    FF_WTA_clean = np.array([resp(x) for x in np.loadtxt(data + "full_field_WTA_clean.csv")])
    
    FF_WTA_noisy = np.reshape(FF_WTA_noisy,[5,model_params["ntrial"],len(con),32,2,model_params['nang']])
    FF_WTA_clean = np.reshape(FF_WTA_clean,[5,1,len(con),32,2,model_params['nang']])
    
    FFW_n_mean = FF_WTA_noisy.mean(axis = 1)
    FFW_n_SD = FF_WTA_noisy.std(axis = 1)
    
    FFW_c_mean = FF_WTA_clean.mean(axis = 1)
    FFW_c_SD = FF_WTA_clean.std(axis = 1)

    for x in FFW_n_mean[0,0,:]:
        print(x[:,0])

    for x in FFW_n_mean[0,-1,:]:
        print(x[:,0])

    ##HWHH

    #ORI TUNING with contrast

    for k in range(len(FFW_n_mean)):
        fn += 1
        plt.figure(fn)
        RL = 0

        I2 = np.argmin(np.abs(con - .25))

        tempmax = np.max(np.append(FFW_n_mean[k,-1,:,0,0],FFW_n_mean[k,-1,-1,0,0]))

        plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
                     np.roll(np.append(FFW_n_mean[k,-1,:,0,0],FFW_n_mean[k,-1,-1,0,0]),RL)/tempmax
                     ,"k")

        tempmax = np.max(np.append(FFW_n_mean[k,I2,:,0,0],FFW_n_mean[k,I2,-1,0,0]))
        plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
                     np.roll(np.append(FFW_n_mean[k,I2,:,0,0],FFW_n_mean[k,I2,-1,0,0]),RL)/tempmax
                     ,"k--")
            
        plt.xticks(np.linspace(0,180,5))
        plt.ylim(0,1)
        plt.tight_layout()

        plt.savefig(data+"ORI_TUN_{}".format(k)+ext)

    for k in range(len(FFW_n_mean)):
        fn += 1
        plt.figure(fn)
        
        plt.xlabel("Target Contrast")
        plt.ylabel("Mask Contrast")
        
        II = np.array([-1,3*len(con)/4,len(con)/2,len(con)/4])

        for c1 in range(len(II)):
            
            plt.subplot(2,2,c1+1)
            RL = int(len(FFW_c_mean[k,0])/4)
            

#            plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
#                     (
#                     np.roll(np.append(FFW_n_mean[k,II[c1],:,0,0],FFW_n_mean[k,II[c1],-1,0,0]),RL)
#                     +
#                     np.roll(np.append(FFW_n_mean[k,0,:,1,0],FFW_n_mean[k,0,-1,1,0]),RL)
#                     )
#                     ,"k--")
            
            plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
                     np.roll(np.append(FFW_n_mean[k,II[c1],:,0,0],FFW_n_mean[k,II[c1],-1,0,0]),RL)
                     ,"g")
            
            plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),
                     np.roll(np.append(FFW_n_mean[k,0,:,1,0],FFW_n_mean[k,0,-1,1,0]),RL)
                     ,"b")

            plt.plot(np.linspace(0,180,len(FFW_n_mean[k,0]) + 1),np.roll(np.append(FFW_n_mean[k,II[c1],:,1,0],FFW_n_mean[k,II[c1],-1,1,0]),RL),"k")
            plt.xticks(np.linspace(0,180,3))

            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off

            plt.tight_layout()

        plt.tight_layout()


        plt.savefig(data+"COS_WTA_{}".format(k)+ext)
            
    for k in range(len(FFC_n_mean)):
        for d in [-1]:
            fn += 1
            plt.figure(fn)
            
            temp = np.array((FFC_n_mean[k,d,:,:,2,0] + np.transpose(FFC_n_mean[k,d,:,:,2,0]))/2)
            CS = plt.pcolor(np.array(con),np.array(con),temp,label = "A.I.")
            
            plt.colorbar()

            CS = plt.contour(np.array(con),np.array(con),temp,label = "A.I.",colors = 'k',linewidths = 1)
            
            plt.xlabel("Target Contrast")
            plt.ylabel("Mask Contrast")
            
            plt.tight_layout()
            plt.axes().set_aspect('equal', 'datalim')

            plt.savefig(data+"COS_CON_{}".format(k)+ext)

            fn += 1
            plt.figure(fn)
            
            temp = np.array((FFC_c_mean[k,d,:,:,2,0] + np.transpose(FFC_c_mean[k,d,:,:,2,0]))/2)
            CS = plt.pcolor(np.array(con),np.array(con),temp,label = "A.I.")
            
            plt.colorbar()

            CS = plt.contour(np.array(con),np.array(con),temp,label = "A.I.",colors = 'k',linewidths = 1)
            
            plt.xlabel("Target Contrast")
            plt.ylabel("Mask Contrast")
            
            plt.tight_layout()
            plt.axes().set_aspect('equal', 'datalim')

            plt.savefig(data+"nnCOS_CON_{}".format(k)+ext)

            
            fn += 1
            plt.figure(fn)
            
            temp = np.array((FFC_n_mean[k,d,:15,:15,2,0] + np.transpose(FFC_n_mean[k,d,:15,:15,2,0]))/2)
            CS = plt.pcolor(np.array(con[:15]),np.array(con[:15]),temp,label = "A.I.")
            
            plt.colorbar()

            CS = plt.contour(np.array(con[:15]),np.array(con[:15]),temp,label = "A.I.",colors = 'k',linewidths = 1)
            
            plt.xlabel("Target Contrast")
            plt.ylabel("Mask Contrast")
            
            plt.tight_layout()
            plt.axes().set_aspect('equal', 'datalim')

            plt.savefig(data+"COS_CON_zoomed_{}".format(k)+ext)
            
            
    for k in range(len(FFC_n_mean)):
        for d in [-1]:
            fn += 1
            plt.figure(fn)

            plt.plot(np.array(con),FFC_n_mean[k,d,:,15,0,0],'r',label = 'noisy')
            plt.plot(np.array(con),FFC_c_mean[k,d,:,15,0,0],'r--',label = "noiseless")

            plt.plot(np.array(con),FFC_n_mean[k,d,:,15,2,0],'k')
            plt.plot(np.array(con),FFC_c_mean[k,d,:,15,2,0],'k--')
                        
            plt.xlabel("Target Contrast")
            plt.ylabel("Response")

            plt.xscale("log")
            
            plt.tight_layout()

            plt.savefig(data+"COS_CRF_{}".format(k)+ext)

    for k in range(len(FFC_n_mean)):
        for d in range(len(FFC_n_mean[k])):
            fn += 1
            plt.figure(fn)
            
            plt.plot(con,[FFAI_n[k,d,i,i,0] for i in range(len(con))],"r",label = "noisy")
            plt.plot(con,[FFAI_c[k,d,i,i,0] for i in range(len(con))],"r--",label = "noiseless")
            plt.plot(con,[1. for i in range(len(con))],"k--")

            if LEG:
                plt.legend()
 
            plt.xlabel("Target Contrast")
            plt.ylabel("Additivity Index")
            
            plt.tight_layout()

            plt.savefig(data+"COS_AI_{}_{}".format(k,d)+ext)


if __name__ == "__main__":
    
    assert len(sys.argv) == 2
    
    data = sys.argv[1] + "/"
    
    print(data.split("center"))
    

    if len(data.split("center")) == 1:
        print("MGSM")
        MGSM_resp(data)
    else:
        print("GSM")
        GSM_resp(data)
