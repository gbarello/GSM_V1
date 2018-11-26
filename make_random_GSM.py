import glob
import numpy as np
import image_processing.image_processing as proc
import scipy.signal as signal
import GSM.MGSM_train as TRAIN
import GSM.MGSM_inference as inf
import image_processing.stimuli as stim

'''
The purpose of this script is to generate the RFs of a random GSM consisting of N layers, in a hexagonal grid, of sites with orientations sampled from a random orientation map. I will center the map to have 
'''

#get the BSDS file location
F = open("./CONFIG","r")
for l in F:
    BSDSloc = l.split("=")[1]
    break
F.close()
###########################

def IQR(x):
    return np.percentile(x,75) - np.percentile(x,25)

def make_ori_map(v_size,n_site,ori_size,n_ori_wave = 30):
    '''
    Description: Makes a sample orientation map

    args: 
    v_size: size of patch of V1 to be simulated
    n_site: number of sites to simulate
    ori_size: scale of orientation map on V1
    n_ori_wave: number of terms to sum for ori map (30 is fine)

    returns: array of values between [0,pi] giving orientation of orientation map
    '''

    
    #for this I want to know the # of sites, the total size (in mm) and size of minicolumns)

    dx = np.array(v_size)/np.array(n_site)
    omap_period = ori_size

    out = np.zeros(n_site)

    rand_o = 2 * np.random.random_integers(0,1,n_ori_wave) - 1
    rand_ph = 2 * np.pi * np.random.random(n_ori_wave)

    for ii in range(n_ori_wave):
        KJ = 2 * np.pi * np.array([np.cos(ii * np.pi / n_ori_wave),np.sin(ii * np.pi / n_ori_wave)]) / omap_period
        out = out + np.exp(1j * np.array([[((x*KJ[0] + y*KJ[1])*rand_o[ii] + rand_ph[ii]) for y in range(n_site)] for x in range(n_site)]))

    return np.angle(out)/2.    

def get_random_gsm_map(Nlayer,ldist,oscale):

    '''
    Description: This function generates a random map of RFs in a hexagonal grid pattern.

    args:
    Nlayer: the number of layers of hexagonal grid
    ldist: distance (iun pixels) between layers in the hex grid.
    oscale: the orientation map scale (relative to hex grid scale)

    returns: [loc,ori,phi]
    loc: location relative to center
    ori: orientation of RF
    phi: phase of RF
    '''
    side = (Nlayer + 1)*ldist

    width = 2*side + 1

    omap = make_ori_map(1.,width,float(oscale)*ldist)

    sites = [[0,0]]

    for k in range(Nlayer):
        corner = []
        for o in range(6):
            corner.append(np.array([np.cos(2*np.pi*o/6)*ldist*(k+1),np.sin(2*np.pi*o/6)*ldist*(k+1)]))

        for c in range(len(corner)):
            sites.append(corner[c])
            for n in range(k):
                sites.append(corner[c] + (n+1)*(corner[np.mod(c+1,len(corner))] - corner[c])/(k+1))

    #retrict the sites to integers
    sites = np.int32(np.array(sites))

    #sample the oris from the map, rotating to align with center
    oris = [omap[sites[k][0]+side,sites[k][1]+side] - omap[side,side] for k in range(len(sites))]

    #get random phases
    phase = np.random.randint(0,2,len(oris))
    #but fix the middle to be even phase
    phase[0] = 0

    return sites,oris,phase

def make_RFs(oris,phase,f,t):
    
    RFs = [proc.LAPc(oris[i],f,t) if phase[i] == 0 else proc.LAPs(oris[i],f,t) for i in range(len(oris))]

    return np.array(RFs)

def make_image_data(im,sites,RF):
    scale = int(np.linalg.norm(sites[2]))
    
    maps = np.array([signal.convolve(im,r,mode = "valid") for r in RF])

    XX = [np.min(sites[:,0]),np.max(sites[:,0])]
    YY = [np.min(sites[:,1]),np.max(sites[:,1])]

    print("grabbing data")

    data = np.array([[maps[m,x + sites[m,0],y + sites[m,1]] for m in range(len(maps))] for x in range(-XX[0],len(maps[0])-XX[1],scale) for y in range(-YY[0],len(maps[0,x])-YY[1],scale)])

    return data

def get_grating_data(im,sites,RF):
        
    maps = np.array([signal.convolve(im,r,mode = "valid") for r in RF])

    cen = (np.array(maps[0].shape) - 1)/2

    data = [maps[m,cen[0]+sites[m,0],cen[1]+sites[m,1]] for m in range(len(sites))]

    return data

def get_random_GSM(Nlayer,ldist,oscale,f,t):

    sites,oris,phase = get_random_gsm_map(Nlayer,ldist,oscale)

    sites = np.array([s for s in sites for k in range(2)])    
    oris = np.array([s for s in oris for k in range(2)])  
    phase = np.array([k for s in oris for k in [0,1]])

    RF = make_RFs(oris,phase,f,t)    
    print(RF.shape)
    
    imlist = glob.glob(BSDSloc + "*.jpg")

    np.random.shuffle(imlist)
    
    Clist = []
    
    for i in imlist:
        Clist.append(make_image_data(proc.load_grayscale_image(i),sites,RF))
        print(i,Clist[-1].shape)
        
    #we want to sample from each one equally, so we find the list with the fewest entries
    mlen = min([len(c) for c in Clist])
    
    #randomise the list and cocnatenate them all into one list
    Clist = np.array([c[np.random.choice(range(len(c)),mlen)] for c in Clist])
    Clist = np.array([k for c in Clist for k in c])
    
    fac = np.array([IQR(Clist[:,k]) for k in range(len(Clist[0]))])
    
    Clist = Clist / np.array([fac])
        
    #randomize before fit
    np.random.shuffle(Clist)
    
    print("Number of samples: {}".format(Clist.shape[0]))
    print("Mean : {} std : {}".format(np.median(Clist),np.std(Clist)))
    print("Max : {}".format(np.max(np.reshape(Clist,[-1]))))
    print("data shape : {}".format(Clist.shape))

    #now I need to run the EM algorithm 
    
    CNS,record = TRAIN.fit_GSM(Clist)

    return CNS,fac,sites,oris,phase,RF
    
if __name__ == "__main__":
    
    oris = []

    for sp in np.linspace(.1,10,30):
        print(sp)
        for k in range(100):
            site,ori,phase = get_random_gsm_map(4,10,sp)

            oris.append([sp] + list(ori))

    np.savetxt("./ori_map_test.csv",oris)

    exit()
            
            
    CNS,fac,sites,oris,phase,RF = get_random_GSM(2,16,.25,16,16*5)
    
    grat = stim.make_SS_filters(.5,0,0,0,16,0,5*16,int(np.max(np.linalg.norm(sites,axis = 1))),get_grat = True)
    
    inp = np.array([get_grating_data(g,sites,RF)/fac for g in grat])


    nc = CNS
    f = .5*np.identity(len(CNS))
    q = inf.Q_self_con(CNS,f)

    inp += np.random.multivariate_normal(np.zeros_like(inp[0]),nc,len(inp))

    aI = np.tile(np.expand_dims(inp,1),[1,3,1])
    
    resp = inf.gnn(inp,CNS)
    iresp = inf.att_gexp(0,aI,CNS,nc,q,f)

    f = .75*np.identity(len(CNS))
    q = inf.Q_self_con(CNS,f)

    aresp = inf.att_gexp(0,aI,CNS,nc,q,f)

    print(oris)

    np.savetxt("./random_gsm_resp.csv",resp)
    np.savetxt("./a_random_gsm_resp.csv",aresp)
    np.savetxt("./i_random_gsm_resp.csv",iresp)
