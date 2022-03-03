index = list(range(0,data.shape[1]-1,int(1/rescale_factor)))
    index = ['TS_'+ str(i) for i in index]+['CLASS']
    data = data[index]
    index_rename = list(range(0,data.shape[1]-1))
    index_rename = ['TS_'+ str(i) for i in index_rename]+['CLASS']
    data.columns = index_rename
    data.to_csv('../data/physionet_A_N_rescaled.csv',index=False)