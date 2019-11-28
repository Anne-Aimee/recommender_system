import seaborn as sns
import pandas as pd
def heatmap_matrix(train, test,lambda_user,lambda_item) :
    heatmapmtrix=np.zeros((lambda_user.shape[0],lambda_item.shape[0]))
    for i in range(lambda_user.shape[0]): 
        for j in range(lambda_item.shape[0]):
            ij= matrix_factorization_SGD_regularized(train,test,lambda_user[i],lambda_item[j])
            heatmapmatrix[i][j]=ij
    return heatmapmatrix

def plot_heatmap(train, test,lambda_user,lambda_item) :
    df=pd.DataFrame(heatmap_matrix(train, test,lambda_user,lambda_item))
    sns.heatmap(df, annot=True, annot_kws={"size": 7})
    
    lambda_user=np.array([0.8])
lambda_item=np.array([0.8])
plot_heatmap(train,test,lambda_user,lambda_user)