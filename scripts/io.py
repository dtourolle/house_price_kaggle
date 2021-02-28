import yaml
import pandas as pd 
import numpy as np

cat_to_ord={'nan':0,'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5, 'No':1,'Mn':2,'Av':3,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}

def Ordinal(data,name):
    out=np.array([cat_to_ord.get(d,np.NaN) for d in data])
    out[np.isnan(out)]=np.median(out)
    return out

def OneHot(column,name):
    temp = pd.get_dummies(column)

    return {name+'_'+str(col):temp[col].values for col in temp }

def numerical(data,name):
    return np.array(data)


def load_dataset(file="../data/train.csv",description='../data/data_desription.yaml',methods={'OneHot':OneHot,'Ordinal':Ordinal,None:numerical}):

    with open(description, 'r') as stream:   # 'document.yaml' contains a single YAML document.
        meta=yaml.load(stream)

    data = pd.read_csv(file)#to_numeric 
    data.sort_values(by=['YrSold'], inplace=True, ascending=True)

    new_data={}
    for name,m in meta.items():

        if m.get('type') == 'categorical':
            d = methods.get(m.get('method'))(data[name],name)
            if type(d) == dict:
                
                new_data = {**new_data, **d}
            else:
                new_data[name]=d

            
        if m.get('type') == 'numerical':
            new_data[name]=np.array(pd.to_numeric(data[name]))
        if m.get('NaNis') == 'median':
            print(name)
            
            new_data[name][np.isnan(new_data[name])] = np.median(new_data[name][np.isnan(new_data[name])==False])
        if m.get('NaNis') == 'mean':
            new_data[name]=np.array(pd.to_numeric(data[name]))
            new_data[name][np.isnan(new_data[name])] = np.mean(new_data[name][np.isnan(new_data[name])==False]) 
        if m.get('NaNis') == 0:
            new_data[name]=np.array(pd.to_numeric(data[name]))
            new_data[name][np.isnan(new_data[name])] = 0

    return new_data