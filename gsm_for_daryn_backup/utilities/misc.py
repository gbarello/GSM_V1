
def get_parameters(name):
#parameters: this just checks the parameter inputs
    if len(name) > 1:
        params = name[1].split("_")
        if len(params) != 9:
            print("'" + name + "' is not a valid parameter specification\n")
            print("parameter specification must be 8 entries long: {1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}")
            print("\t{1} - dataset : 'BSDS' or 'none'.")
            print("\t{2} - wavelength of gabor filters in pixels.")
            print("\t{3} - scale of RF in pixels.")
            print("\t{4} - total size of convolutions in pixels (at least 5*scale recommended).")
            print("\t{5} - number of filter positions in surround.")
            print("\t{6} - number of orientations to sample.")
            print("\t{7} - number of phases to sample.")
            print("\t{8} - distance in pixels between center and surround.")
            print("\t{9} - distance in pixels between samling points in images.\n")
            exit()
            
        dataset = params[0]
        freq = int(params[1])
        scale = int(params[2])
        tot = int(params[3])
        nfilt = int(params[4])
        nang = int(params[5])
        npha = int(params[6])
        fdist = int(params[7])
        samd = int(params[8])
        
    else:
        
        freq = 5
        scale = 5
        tot = 6*scale
        nfilt = 8
        nang = 4
        fdist = 20
        npha = 2
        samd = 10
        dataset = "BSDS"
        
    return dataset,freq,scale,tot,nfilt,nang,npha,fdist,samd
