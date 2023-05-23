import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import matrix_rank

def LI_vecs(dim,M):
    LI=[M[0]]
    idxs=[0]
    for i in range(dim):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])                #set tmp=LI+[M[i]]
        if matrix_rank(tmp)>len(LI):    #test if M[i] is linearly independent from all (row) vectors in LI
            LI.append(M[i])             #note that matrix_rank does not need to take in a square matrix
            idxs.append(i)
    return idxs

def find_independent_columns(exog):
    if np.linalg.matrix_rank(exog)==np.shape(exog)[1]:
        return np.array(exog.columns)
    idxs = LI_vecs(np.shape(exog)[1], exog.T.values)
    return np.array(exog.columns[idxs])

def normalize_features(df_feature, features_to_norm):
    scaler = StandardScaler()
    for f in features_to_norm:
        df_feature[f] = scaler.fit_transform(df_feature[f].values.reshape(-1, 1)).flatten()
    return df_feature

# https://stackoverflow.com/a/21739593
def correct_pvalues_for_multiple_testing(pvalues, correction_type = "Benjamini-Hochberg"):                
    """                                                                                                   
    consistent with R - print correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05, 0.069, 0.07, 0.071, 0.09, 0.1]) 
    """
    from numpy import array, empty
    pvalues = array(pvalues) 
    n = pvalues.shape[0]                                                           
    new_pvalues = np.zeros(n)
    if correction_type == "Bonferroni":                                                                   
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":                                                            
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        for rank, vals in enumerate(values):                                                              
            pvalue, i = vals
            new_pvalues[i] = (n-rank) * pvalue                                                            
    elif correction_type == "Benjamini-Hochberg":                                                         
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]                                      
        values.sort()
        values.reverse()                                                                                  
        new_values = []
        for i, vals in enumerate(values):                                                                 
            rank = n - i
            pvalue, index = vals                                                                          
            new_values.append((n/rank) * pvalue)                                                          
        for i in range(0, int(n)-1):  
            if new_values[i] < new_values[i+1]:                                                           
                new_values[i+1] = new_values[i]                                                           
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]                                                                                                                  
    return new_pvalues