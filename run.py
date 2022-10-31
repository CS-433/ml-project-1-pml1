# +
from helpers import *
from implementations import *
from Data_Cleaning import *
import pickle


def main(path_submission = "new_submission.csv"):
    
    y, X, ids = load_csv_data("data/test.csv", sub_sample=False)
    
    w_0 = np.load('save_param/best_w_0_M3_for_0_1_2.npy',allow_pickle=True)
    w_1 = np.load('save_param/best_w_1_M3_for_0_1_2.npy',allow_pickle=True)
    w_2 = np.load('save_param/best_w_2_default_indicatrice.npy',allow_pickle=True)
    
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
    
    with open('save_param/information_transformation_final.json', 'rb') as fp:
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
    
    create_csv_submission(ids, y, "new_submission.csv")
    
    print(f'File new_submission.csv created')
    
    return 

   

if __name__ == '__main__':
    main()
    
