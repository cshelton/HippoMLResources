import numpy as np
import json
import gzip

# simple dataset class
# just directly access and use (basically, just standardizes the field names)
class dataset:
    def __init__(self):
        self.X = None
            # X is an m-by-n numpy array
        self.Y = None
            # Y is None if this is an unsupervised learning dataset
        self.featnames = []
            # featnames[i] is the name of feature x_i
        self.featcats = []
            # featcats[i][j] is the name of the jth value for feature x_i
            # featcats[i] is None if feature x_i is real valued 
        self.yname = None
            # yname is the name of the target
        self.ycats = None
            # ycats is None if y is real valued, otherwise it is
            # a list of the names of the possible values
        self.attribution = None
            # attribution for data

    # saves dataset into two files:
    # <filestem>-XY.pyz  and <filestem>-meta.json
    # with the former containing the X and Y numpy arrays
    # and the latter containing the other meta data
    def save(self,filestem,compressmeta=False):
        with open(filestem+'-XY.pyz','wb') as fxy:
            if self.X is not None:
                if self.Y is not None:
                    np.savez_compressed(fxy,X=self.X,Y=self.Y,allow_pickle=False)
                else:
                    np.savez_compressed(fxy,X=self.X,allow_pickle=False)
            elif self.Y is not None:
                np.savez_compressed(fxy,Y=self.Y,allow_pickle=False)
            else:
                np.savez_compressed(fxy,ignore=[],allow_pickle=False)

        if compressmeta:
            with gzip.open(filestem+'-meta.json.gz','wt',encoding='UTF-8') as fmeta:
                json.dump(dict(featnames=self.featnames,
                               featcats=self.featcats,
                               yname=self.yname,
                               ycats=self.ycats,
                               attribution = self.attribution),
                          fmeta,
                          separators=(',',':'))
        else:
            with open(filestem+'-meta.json','w') as fmeta:
                json.dump(dict(featnames=self.featnames,
                               featcats=self.featcats,
                               yname=self.yname,
                               ycats=self.ycats,
                               attribution = self.attribution),
                          fmeta,
                          separators=(',',':'))

# loads (and returns) a dataset from the files
# <filestem>-XY.pyz and <filestem>-meta.json
def loaddataset(filestem):
    ret = dataset()
    with open(filestem+'-XY.pyz','rb') as fxy:
        xy = np.load(fxy)
        if 'X' in xy:
            ret.X = xy['X']
        if 'Y' in xy:
            ret.Y = xy['Y']

    try:
        with open(filestem+'-meta.json','r') as fmeta:
            meta = json.load(fmeta)
            for field in ['featnames','featcats','yname','ycats','attribution']:
                if field in meta:
                    setattr(ret,field,meta[field])
    except FileNotFoundError:
        with gzip.open(filestem+'-meta.json.gz','rt',encoding='UTF-8') as fmeta:
            meta = json.load(fmeta)
            for field in ['featnames','featcats','yname','ycats','attribution']:
                if field in meta:
                    setattr(ret,field,meta[field])

    return ret
