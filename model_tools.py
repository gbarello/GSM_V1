import numpy as np
import utilities as utils

def get_mean_NC(C,segs):

    NN = len(np.concatenate(segs[0]))

    COUNT = np.zeros([NN,NN])

    NC = np.zeros([NN,NN])

    for s in range(len(segs)):
        for a in range(len(segs[s])):
            for k in range(len(segs[s][a])):
                for j in range(len(segs[s][a])):
                    NC[segs[s][a][k],segs[s][a][j]] += C[s][a][k,j]
                    COUNT[segs[s][a][k],segs[s][a][j]] += 1

    for a in range(len(COUNT)):
        for b in range(len(COUNT[a])):
            if COUNT[a][b] == 0:
                COUNT[a][b] = 1
           
    out = NC/len(segs)
    assert np.all(out == out.transpose())
    return out

def get_kernel_NC(kernels,fpos,fac):

    '''
    This function expects a 3-D array kernels which has the kernels for each gaussian variable and a 3-D array fpos which has the x,y corordinates of the kernel relative to zero for all teh kernels. It also expects a 1D array fac which stores the factors by which outputs are scaled in the model
    
    it returns their inner product, normalized w.r.t. fac
    '''
    
    pad = np.int32(np.max(np.abs(fpos)))

    ret = [[np.mean(pad_kernel(kernels[i],np.int32(fpos[i]),pad)*pad_kernel(kernels[j],np.int32(fpos[j]),pad)/(fac[i]*fac[j])) for j in range(len(kernels))] for i in range(len(kernels))]

    return np.array(ret)
    
def pad_kernel(k,p,pad):

    temp = np.zeros([k.shape[0]+2*pad,k.shape[1]+2*pad])
    temp[pad+p[0]:pad+p[0]+k.shape[0],pad+p[1]:pad+p[1]+k.shape[1]] = k

    return temp
    
def get_model_data(direc):

    fac = np.loadtxt(direc + "/fac.csv")
    log = np.loadtxt(direc + "/train_log.csv")
    param = utils.read_dict(direc + "/parameters")
    paths = utils.fetch_file(direc + "/paths.pkl")
    segs = utils.fetch_file(direc + "/segs.pkl")
    kernels = utils.fetch_file(direc + "/kernels.pkl")

    C = utils.fetch_file(direc + "/C.pkl")
    Q = utils.fetch_file(direc + "/Q.pkl")
    F = utils.fetch_file(direc + "/F.pkl")
    P = utils.fetch_file(direc + "/P.pkl")

    return {"fac":fac,"params":param,"paths":paths,"kernels":kernels,"C":C,"Q":Q,"F":F,"P":P,"segs":segs,"log":log}

def flat(x):
    return np.reshape(x,[-1]).tolist()

def get_f_pos(pos_type,fdist,nsur):

    if pos_type == "default":
        return [[0,0]] + [[fdist*np.cos(a),fdist*np.sin(a)] for a in np.linspace(0,np.pi*2,nsur+1)][:-1]
    elif pos_type == "extended":
        return [[0,0]] + [[fdist*np.cos(a)/2,fdist*np.sin(a)/2] for a in np.linspace(0,np.pi*2,4+1)][:-1] + [[fdist*np.cos(a),fdist*np.sin(a)] for a in np.linspace(0,np.pi*2,nsur+1)][:-1]
    elif pos_type == "line":
        return [[fdist*a,0] for a in np.linspace(0,nsur/2,nsur + 1)]

def get_segmentation(seg_type,nang,wave,fpos):

    if seg_type == "gsm":
        #only 1 seg. All the finters are in it.
        indices = [[[i for i in range(nang*len(wave)*len(fpos)*2)]]]
        return indices

    elif seg_type == "in-out":
        #2 seg. one with cen-sur segmented, the other with them together
        out = [[i for i in range(nang*len(wave)*2)],[i for i in range(nang*len(wave)*2,nang*len(wave)*2*len(fpos))]]

        allin = [[i for i in range(nang*len(wave)*len(fpos)*2)]]

        return [out,allin]
    
    elif seg_type == "line":
        out = [[[i for i in range(0,2*len(wave)*nang*k)],[i for i in range(2*len(wave)*nang*k,nang*len(wave)*2*len(fpos))]] for k in range(1,len(fpos) - 1)] + [[[i for i in range(0,2*len(wave)*nang*len(fpos))]]]

        return out

    elif seg_type == "default":
        #the default (CC) is to have a non-shared seg (center separate from surrounds) plus one seg for each angle, with the center shared with the surround of that angle, and other amgles separate.

        
        indices = []
        
        i = 0
        for p in fpos:
            for a in range(nang):
                for w in wave:
                    for t in range(2):
                        indices.append(i)
                        i+=1

        indices = np.reshape(np.array(indices),[len(fpos),nang,len(wave),2])

        seg = []

        seg.append([flat(indices[0])] + [flat(indices[1:,i]) for i in range(nang)])
        
        for i in range(nang):
            seg.append([flat(indices[0]) + flat(indices[1:,i])] + [flat(indices[1:,k]) for k in range(nang) if k != i])

        return seg

    elif seg_type == "extended":
        #the extended is to have a non-shared seg (center separate from surrounds) plus one seg for each angle, with the center shared with the surround of that angle, and other amgles separate, but have two layers of surround, with segmentations the have center nad first surround, adn center and both surrounds. 

        
        indices = []
        
        i = 0
        for p in fpos:
            for a in range(nang):
                for w in wave:
                    for t in range(2):
                        indices.append(i)
                        i+=1

        #the expectation is that there are 4 middle-surrounds and 8 far surrounds 
        indices = np.reshape(np.array(indices),[len(fpos),nang,len(wave),2])

        seg = []

        seg.append([flat(indices[0])] + [flat(indices[1:,i]) for i in range(nang)])

        for i in range(nang):
            seg.append([flat(indices[0]) + flat(indices[1:5,i])] + [flat(indices[1:,k]) if k != i else flat(indices[5:,k]) for k in range(nang)])

        for i in range(nang):
            seg.append([flat(indices[0]) + flat(indices[1:,i])] + [flat(indices[1:,k]) for k in range(nang) if k != i])

        return seg
    else:
        print("segmentation not recognized.")
        exit()

if __name__ == "__main__":
    if 0:
        from GSM.MGSM_inference import check_seg
    
        A = get_segmentation("default",4,[16],[i for i in range(9)])
        
        for a in A:
            print(len([x for B in a for x in B]))
            
        print(check_seg(A))

    else:
        print(get_f_pos("default",10,8))
        print(len(get_f_pos("default",10,8)))
