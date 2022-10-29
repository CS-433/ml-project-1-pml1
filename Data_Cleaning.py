from helpers import *
from implementations import *

def filter_data(X,col,angle_col,model,deg_cross_term,frequence,deg_cross_sin,_type_="default"):
    
    angle_col_complement = []
    for c in col:
        if not(c in angle_col):
            angle_col_complement.append(c)
    
    if _type_ == "default":
              
        return np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),
                                                      deg_cross_term+1),
                               build_poly_cross_terms(X[:,[x for x in angle_col_complement if x <= 29]],deg_cross_term),
                              sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), frequence, deg_cross_sin)], axis = 1)
def correctOutliers(xi, lower, upper, newVals):
    a = xi > upper
    b = xi < lower
    aorb = np.logical_or(a,b)
    xxi = np.where(aorb, newVals, xi)
    return xxi

def handleOutliers(tx):
    q3, q2, q1 = np.quantile(tx, [0.75, 0.5, 0.25], axis=0)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    tx2 = np.apply_along_axis(correctOutliers, axis=1, arr=tx, lower=lower, upper=upper, newVals=q2)
    return tx2


def split_according_num_split_jet(y,X,ids):
    """ This method split the dataset into three sub-dataset. One for each case PRI_num_jet = 0, PRI_num_jet = 1,
    PRI_num_jet >= 2, since for each of this case the available features are not the same. Before using this method,
    we should complete the first column of X, since the DER_mass_MMC is not always definedfor each of this case but 
    is not meaningless.
    """
    ind_0 = np.where(X[:,22]==0)
    ind_1 = np.where(X[:,22]==1)
    ind_2 = np.where(X[:,22]>=2)
    
    # features to keep for each model
    col_0 = [0,1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,29,30,
             31,32,33,34,35,36,37,38,39,40,41,42,43,44]
    col_1 = [0,1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,
             25,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
             47,48,49]
    col_2 = np.arange(0,55,1)
    
    angle_0 = [15,18,20,43,44]
    angle_1 = [15,18,20,25,43,44,49]
    angle_2 = [15,18,20,28,25,43,44,49,54]
    
    X_0 = filter_data(X[ind_0],col_0,angle_0,0,3,[1,2],2)
    
    X_1 = filter_data(X[ind_1],col_1,angle_1,1,3,[1,2],2)
    
    X_2 = np.concatenate([filter_data(X[ind_2],col_2,angle_2,2,2,[1,2],2),
                     (X[ind_2, 22] == 2).astype(np.float64).T,
                     (X[ind_2, 22] == 3).astype(np.float64).T],axis = 1)
    
    return {"0": {"data": X_0,
                  "label": y[ind_0],
                  "ids": ids[ind_0],
                  "ind": ind_0},
           "1": {"data": X_1,
                  "label": y[ind_1],
                  "ids": ids[ind_1],
                  "ind": ind_1},
           "2": {"data": X_2,
                  "label": y[ind_2],
                  "ids": ids[ind_2],
                  "ind": ind_2}}

def DER_mass_MMC_completion(type_,X):
    if type_ == "mean":
        X[:,0][X[:,0]==-999] = np.mean(X[:,0][X[:,0]!=-999])
        return X
    