from helpers import *
from implementations import *
import pickle


def prediction_on_test_set():
    
    y, X, ids = load_csv_data("test.csv", sub_sample=False)
    
    w_0 = np.load('best_w_0_M3_for_0_1_2.npy',allow_pickle=True)
    w_1 = np.load('best_w_1_M3_for_0_1_2.npy',allow_pickle=True)
    w_2 = np.load('best_w_2_default_indicatrice.npy',allow_pickle=True)
    
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
    
    #information_normalisation = np.load('information_transformation_final.npz',allow_pickle=True)
    with open('information_transformation_final.json', 'rb') as fp:
        information_normalisation = pickle.load(fp)
    
    
    X_0 = filter_data(X[ind_0],col_0,angle_0,0,_type_ = "M3",prediction = True, 
                      mean_transformation = information_normalisation["0"]["mean_log"],
               sigma_transformation = information_normalisation["0"]["sigma_log"], 
                      mean = information_normalisation["0"]["mean"], 
                      sigma = information_normalisation["0"]["sigma"], median = information_normalisation["0"]["median"])
    
    X_1 = filter_data(X[ind_1],col_1,angle_1,1,_type_ = "M3",prediction = True, 
                      mean_transformation = information_normalisation["1"]["mean_log"],
               sigma_transformation = information_normalisation["1"]["sigma_log"], 
                      mean = information_normalisation["1"]["mean"], 
                      sigma = information_normalisation["1"]["sigma"], median = information_normalisation["1"]["median"])
    
    
    X_2 = np.concatenate([X[ind_2],(X[ind_2, 22] == 2).astype(np.float64).T,
                              (X[ind_2, 22] == 3).astype(np.float64).T],axis = 1)
    X_2 = filter_data(X_2,col_2,angle_2,2,prediction = True, 
                      mean_transformation = information_normalisation["2"]["mean_log"],
               sigma_transformation = information_normalisation["2"]["sigma_log"], 
                      mean = information_normalisation["2"]["mean"], 
                      sigma = information_normalisation["2"]["sigma"], median = information_normalisation["2"]["median"])
    
        
    
    nan = np.array([not np.isnan(x) for x in np.mean(X_0,axis = 0)])
    
    X_0 = X_0[:,np.where((np.std(X_0,axis=0)!=0)*nan)[0]]
    X_1 = X_1[:,np.where(np.std(X_1,axis=0)!=0)[0]]
    X_2 = X_2[:,np.where(np.std(X_2,axis=0)!=0)[0]]
    
    mean_0 = information_normalisation["0"]["mean"]
    sigma_0 = information_normalisation["0"]["sigma"]
    mean_1 = information_normalisation["1"]["mean"]
    sigma_1 = information_normalisation["1"]["sigma"]
    mean_2 = information_normalisation["2"]["mean"]
    sigma_2 = information_normalisation["2"]["sigma"]
    
    X_0 = (X_0-mean_0)/sigma_0
    X_1 = (X_1-mean_1)/sigma_1
    X_2 = (X_2-mean_2)/sigma_2
    
    y[ind_0]  = predict(w_0, X_0)
    y[ind_1]  = predict(w_1, X_1)
    y[ind_2]  = predict(w_2, X_2)
    
    return y,ids
    
    
    
    
    

def filter_data(X,col,angle_col,model,_type_="default",standardize = True, prediction = False, mean_transformation = None,
               sigma_transformation = None, mean = None, sigma = None, median = None):
    
    if not prediction:
        median = np.median(X[X[:,0]!=-999,0])
        X[X[:,0]==-999,0] = median

        angle_col_complement = []
        for c in col:
            if not(c in angle_col):
                angle_col_complement.append(c)

        if _type_ == "default":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                m= np.mean(X_,axis=0)
                sigma = np.std(X_,axis=0)
                X_ =(X_ - m)/sigma
            return [np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [1,2], 1)], axis = 1),
                                    m, sigma,median]
        elif _type_ == "M1":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                m= np.mean(X_,axis=0)
                sigma = np.std(X_,axis=0)
                X_ =(X_ - m)/sigma
            return [np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),3),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [0.5,1,2], 1)], axis = 1),
                                    m, sigma,median]
        elif _type_ == "M2":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                m= np.mean(X_,axis=0)
                sigma = np.std(X_,axis=0)
                X_ =(X_ - m)/sigma
            return [np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [0.25,0.5,1,2], 2)], axis = 1),
                                    m, sigma,median]
        elif _type_ == "M3":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                m= np.mean(X_,axis=0)
                sigma = np.std(X_,axis=0)
                X_ =(X_ - m)/sigma
            return [np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2,True,13),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3,True,13),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [1,2], 2,True,5)], axis = 1),
                                    m, sigma,median]
    
    else:
        
        X[X[:,0]==-999,0] = median

        angle_col_complement = []
        for c in col:
            if not(c in angle_col):
                angle_col_complement.append(c)

        if _type_ == "default":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                X_ =(X_ - mean_transformation)/sigma_transformation
            return np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [1,2], 1)], axis = 1)
        elif _type_ == "M1":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                X_ =(X_ - mean_transformation)/sigma_transformation
            return np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),3),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                                 sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [0.5,1,2], 1)], axis = 1)
        elif _type_ == "M2":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                X_ =(X_ - mean_transformation)/sigma_transformation
            return np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3),
                            sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [0.25,0.5,1,2], 2)], axis = 1)
        elif _type_ == "M3":
            X_ = log_transformation(two_mode_data(X,col),col)
            if standardize:
                X_ =(X_ - mean_transformation)/sigma_transformation
            return np.concatenate([build_poly_cross_terms(build_derived_quantities(X,model,concatenate=False),2,True,13),
                                   build_poly_cross_terms(X_[:,[x for x in angle_col_complement if x <= 29]],3,True,13),
                            sin_cos(build_add_minus_term(X[:,[x for x in angle_col if x <= 29]]), [1,2], 2,True,5)], axis = 1)

def log_transformation(X,col):
    col_skewed = [0,1,2,3,4,5,8,9,10,13,16,19,21,23,26,29]
    new_col = []
    for i in col_skewed:
        if i in col:
            new_col.append(i)
    X[:,new_col] = np.log(X[:,new_col]-np.min(X[:,new_col],axis=0)+1)
    return X

def two_mode_data(X,col):
    col_mode = [11,12]
    crit_val = [-0.356,0.454]
    new_col = []
    new_crit = []
    for i,c in enumerate(col_mode):
        if c in col:
            new_col.append(c)
            new_crit.append(crit_val[i])
    inf = 1*(X[:,new_col]<new_crit).reshape((X.shape[0],len(new_col)))
    return np.concatenate([X,inf],axis = 1)

def split_according_num_split_jet(y,X,ids, model_2_indicatrice = True):
    
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
    
    
    
    X_0, m_0, s_0, median_0 = filter_data(X[ind_0],col_0,angle_0,0,_type_ = "M3")
    
    X_1, m_1, s_1, median_1 = filter_data(X[ind_1],col_1,angle_1,1,_type_ = "M3")
    
    if model_2_indicatrice:
        X_2 = np.concatenate([X[ind_2],(X[ind_2, 22] == 2).astype(np.float64).T,
                              (X[ind_2, 22] == 3).astype(np.float64).T],axis = 1)
        X_2, m_2, s_2, median_2 = filter_data(X_2,col_2,angle_2,2)
    else:
        X_2, m_2, s_2, median_2 = filter_data(X[ind_2],col_2,angle_2,2)#,_type_ = "M3")
        X_2 = np.concatenate([X_2,(X[ind_2, 22] == 2).astype(np.float64).T,(X[ind_2, 22] == 3).astype(np.float64).T],axis = 1)
    
    nan = np.array([not np.isnan(x) for x in np.mean(X_0,axis = 0)])
    
    X_0 = X_0[:,np.where((np.std(X_0,axis=0)!=0)*nan)[0]]
    X_1 = X_1[:,np.where(np.std(X_1,axis=0)!=0)[0]]
    X_2 = X_2[:,np.where(np.std(X_2,axis=0)!=0)[0]]
    
    mean_0 = np.mean(X_0,axis = 0)
    sigma_0 = np.std(X_0,axis = 0)
    mean_1 = np.mean(X_1,axis = 0)
    sigma_1 = np.std(X_1,axis = 0)
    mean_2 = np.mean(X_2,axis = 0)
    sigma_2 = np.std(X_2,axis = 0)
    
    X_0 = (X_0-mean_0)/sigma_0
    X_1 = (X_1-mean_1)/sigma_1
    X_2 = (X_2-mean_2)/sigma_2
    
    return [{"0": {"data": X_0,
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
                  "ind": ind_2}},
           {"0": {"mean_log": m_0,
                  "sigma_log": s_0,
                  "mean": mean_0,
                  "sigma": sigma_0,
                  "median": median_0},
            "1": {"mean_log": m_1,
                  "sigma_log": s_1,
                  "mean": mean_1,
                  "sigma": sigma_1,
                  "median": median_1},
            "2": {"mean_log": m_2,
                  "sigma_log": s_2,
                  "mean": mean_2,
                  "sigma": sigma_2,
                  "median": median_2},
           }]

def DER_mass_MMC_completion(type_,X):
    if type_ == "mean":
        X[:,0][X[:,0]==-999] = np.mean(X[:,0][X[:,0]!=-999])
        return X
    