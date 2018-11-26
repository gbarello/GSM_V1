import glob
import GSM.MGSM_inference as inf
import numpy as np
import utilities as utils
import model_tools
import image_processing.stimuli as stim
import image_processing.make_dataset as make_data
import GSM.MGSM_inference as inference
import scipy.linalg as linalg
import image_processing.image_processing as proc

def on_off_response(data,SNR,dt,nframes):

    par = data["params"]
    #I want to make a bunch of gratings in 6 orientations
    #I want to simulate them many times, with noise, and record the full temporal response for a while(until stability?)
    #lets go for 100 presentations of each of the 6 gratings.
    #I need enough to do a numerical estimation of the mutual information

    if par["segmentation"] != "gsm":
        print("On off is only for GSM models!")
        exit()

        
    f_pos = model_tools.get_f_pos(par["filter_position"],par["filter_distance"]*np.max(par["wavelengths"]),par["n_surrounds"])
    indices = np.concatenate([[[a,b,c] for a in range(par["n_angles"]) for b in range(len(par["wavelengths"])) for c in range(2)] for p in f_pos])
    positions = np.concatenate([[p for a in range(par["n_angles"]) for b in range(len(par["wavelengths"])) for c in range(2)] for p in f_pos])

    fullsize = int(5*max(par["wavelengths"]) + 2*np.max(f_pos))
    minwav = np.min(par["wavelengths"])
    out = []

    grats = stim.make_grating(.5,0,par["wavelengths"][0],fullsize/2,fullsize)
    print("Getting Coefficients")
    #get filters

    coeffs = make_data.get_filter_maps(grats,data["kernels"])
        
    print("DT: {}".format(dt))
            
    path = [coeffs.shape[-1]/2 for k in range(nframes)]
    print(path)
    rundat = make_data.sample_path(coeffs,path,indices,positions)/np.array([data["fac"]])
    
    print(rundat.shape)
    #add a few zeroes for before and after stimulus onset
    Z = np.zeros([2] + list(rundat.shape[1:]))

    #2 zeros before, 6 after
    rundat = np.concatenate([Z,rundat,Z,Z,Z,Z,Z],axis = 0)

    
    #get all filters
    ind = [k for k in range(len(indices))]
            
    feps = [[linalg.logm(m)/data["params"]["walk_dt"] for m in f] for f in data["F"]]
    FF = [[np.float32(linalg.expm(dt * data["params"]["walk_dt"] * m)) for m in f] for f in feps]
    
    #print(FF[0][0])
    QQ = [[inference.Q_self_con(data["C"][k][m],FF[k][m]) for m in range(len(data["F"][k]))] for k in range(len(data["F"]))]
    NC = [[m/(SNR**2) for m in f] for f in data["C"]]
    
    
    runf = rundat
    
    #run the response analysis
    out = []
    for k in range(1,runf.shape[0]):
        print("{}\t{}".format(k,runf.shape))
        responses = inference.general_MGSM_g_att(np.array([runf[:k]]),data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op = False)
        out.append(responses)

    return np.array(out)

def mutual_information_data(data,dt,snr = [.5,1.],n_samples=1000,use_grat = True):

    #get the BSDS file location
    F = open("./CONFIG","r")
    for l in F:
        BSDSloc = l.split("=")[1]
        break
    F.close()
        
    img_names = glob.glob(BSDSloc + "*.jpg")
    ###########################

    maxframe = 20

    par = data["params"]
    #I want to make a bunch of gratings in 3 orientations
    #I want to simulate them many times, with noise, and record the full temporal response for a while(until stability?)
    #lets go for 100 presentations of each of the 3 gratings.
    #I need enough to do a numerical estimation of the mutual information

    if par["segmentation"] != "gsm":
        print("Mutual Information is only for GSM models!")
        exit()

        
    f_pos = model_tools.get_f_pos(par["filter_position"],par["filter_distance"]*np.max(par["wavelengths"]),par["n_surrounds"])
    indices = np.concatenate([[[a,b,c] for a in range(par["n_angles"]) for b in range(len(par["wavelengths"])) for c in range(2)] for p in f_pos])
    positions = np.concatenate([[p for a in range(par["n_angles"]) for b in range(len(par["wavelengths"])) for c in range(2)] for p in f_pos])


    fullsize = int(5*max(par["wavelengths"]) + 2*np.max(f_pos))
    minwav = np.min(par["wavelengths"])
    out = []
    outv = []
    imnum = 0
    imind = np.random.choice(np.arange(len(img_names)),3,replace = False)

    for o in np.linspace(0,np.pi,4)[:-1]:
        print("ORI: {}".format(o))
        #make the gratings
    
        if use_grat:
            grats = stim.make_grating(1.,o,par["wavelengths"][0],fullsize/2,fullsize)
            print("Getting Coefficients")
            #get filters

        else:
            grats = proc.load_grayscale_image(img_names[imind[imnum]])#.make_grating(1.,o,par["wavelengths"][0],fullsize/2,fullsize)
            imnum += 1
            print("Getting Coefficients")
            #get filters
        coeffs = make_data.get_filter_maps(grats,data["kernels"])

        print(np.array(coeffs).shape)

        #extract the right ones

        out.append([])
        outv.append([])
        
        for n in range(len(snr)):
            out[-1].append([])
            outv[-1].append([])
            print("SNR: {}".format(snr[n]))
            
            path = [coeffs.shape[-1]/2 for k in range(maxframe)]
            print(path)
            rundat = make_data.sample_path(coeffs,path,indices,positions)/np.array([data["fac"]])
            print(rundat.shape)
            #add a few zeroes for before stimulus onset
            Z = np.zeros([int(2)] + list(rundat.shape[1:]))
            rundat = np.concatenate([Z,rundat],axis = 0)
            
            #get all filters
            ind = [k for k in range(len(indices))]
            
            feps = [[linalg.logm(m)/data["params"]["walk_dt"] for m in f] for f in data["F"]]
            FF = [[np.float32(linalg.expm(dt * data["params"]["walk_dt"] * m)) for m in f] for f in feps]
            
            #print(FF[0][0])
            QQ = [[inference.Q_self_con(data["C"][k][m],FF[k][m]) for m in range(len(data["F"][k]))] for k in range(len(data["F"]))]
            NC = [[m/((snr[n])**2) for m in f] for f in data["C"]]

            #add noise
            noise = np.random.multivariate_normal(np.zeros(rundat.shape[1]),NC[0][0],[n_samples,rundat.shape[0]])

            runf = noise + np.expand_dims(rundat,0)

            #run the response analysis
            for k in [1,len(Z)] + range(len(Z)+1,runf.shape[1],2) + [runf.shape[1]]:
                print("{}\t{}".format(k,runf[:,:k].shape))
                responses = inference.general_MGSM_g_att(runf[:,:k],data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op = False)
                vresponses = inference.general_MGSM_p_att(runf[:,:k],data["segs"],data["C"],NC,QQ,FF,data["P"],ind,stable = True,op = False)

                out[-1][-1].append(responses)
                outv[-1][-1].append(vresponses)
            out[-1][-1] = np.array(out[-1][-1])
            outv[-1][-1] = np.array(outv[-1][-1])

    return out,outv
