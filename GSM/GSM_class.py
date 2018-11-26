import sys
sys.path.append('~/data')
sys.path.append('~/my_modules')
import utilities as util
import get_data
from . import MGSM_train as train
from . import MGSM_inference as inference

class response:
    def __init__(self,I,data,label,inf_type,model):
        
        '''
        Description: This class stores and organizes GSM responses. It takes inputs of the filter inputs, a data dict. giving extra parameters, a label to tag the data, and an inference type to perform.

        args:
         I - (array) the filter inputs. Can be any shape so long as I.shape[-1] == CNS.shape[0] == CNS.shape[1]
         data - (dict) a dict of extra data elements. Must include the covariance with key "cov". Can additionally include - covariance, noise covariance, dynamical covariance (for attention), etc.
         label - (str) a tag to label the response with
         inf_type - (str) the type of inference to perform 
        '''
        
        #the data must at least contain a covariance
        assert "cov" in data.keys()

        self.data = data
        self.I = I
        self.label = label
        self.inf_type = inf_type

        if model == "GSM":
            self.g = np.reshape(inference.GSM_gexp(np.reshape(I,[-1,self.I.shape[-1]]),data,inf_type),self.I.shape)
            
        elif model == "MGSM":
            self.g = np.reshape(inference.MGSM_gexp(np.reshape(I,[-1,self.I.shape[-2],self.I.shape[-1]]),data,inf_type),list(self.I.shape[:-2]) + [-1])

class GSM:
    def __init__(self,loadfile = ""):

        if loadfile != "":
            self.load(loadfile)
            
        self.g = []

        self.model_data = {}
        
    def load(self,loc):
        self = util.fetch_file(loc)
        
    def export(self,loc):
        util.dump_file(loc,self)

    def fit_data(self,data_params):
        data = get_data.get_GSM_data(data_params)
        
        cov,test = train.fit_GSM(data)

        self.model_data["data_params"] = data_params
        self.model_data["cov"] = cov

    def response(self,I,fit_type):
        self.g.append(response(I,self.model_data,inf_type,model = "GSM"))

class MGSM:
    def __init__(self,loadfile = ""):

        if loadfile != "":
            self.load(loadfile)
            
        self.g = []

        self.model_data = {}
        
    def load(self,loc):
        self = util.fetch_file(loc)
        
    def export(self,loc):
        util.dump_file(loc,self)

    def fit_data(self,data_params):
        data = get_data.get_MGSM_data(data_params)

        P,cc,cs,ccs= train.fit_MGSM(data)

        self.model_data["data_params"] = data_params
        self.data["cov"] = [cc,cs,ccs]
        self.data["prob"] = P
        
    def response(self,I,inf_type):
        self.g.append(response(I,self.model_data,inf_type,model = "MGSM"))
    
