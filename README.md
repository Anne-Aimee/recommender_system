# Project Recommender System

## Project presentation
The aim of the recommender system project is to predict the appropriate rating of an item to a specific user, based on provided information about these items and their users. Our dataset consists of a series of movies rated by users. These ratings are integer values between 1 and 5 stars, and no additional information is available about the movies or users. Different machine learning (ML) techniques are applied on this known dataset, with the objective to fill its missing entries and guess the unseen rating of a given user for one particular movie.

## Code Structure 
The code is separated in different parts in main.ipynb 
    - Exploring the data
    - Data Analysis
        - Split test and train
        - Baseline predictions
    - Matrix Factorization
        - SGD
        - ALS
        - SGD with baseline correction
        - SGD with non linearity (sigmoid)
    - Model comparison and variance estimation
    - Final prediction
 
### Exploring the data
- number of ratings per user and movies
- bias towards positive ratings
- The number of ratings per movie is correlated with its average rating.
- The same is less true for the user rating.
- Standard deviation of users' ratings

### Data Analysis
- split of test and train data, with selection of "valid data" (>10 ratings /movie and user).
- baselines
    - global mean
    - user mean
    - item mean
    - item mean and user offset with blending. 
    In addition to the item mean, the average offset of the user's rating with respect to the global mean is taken into account. 
    The item mean and the average offset are corrected as a function of the number of counts.
    - simple linear model.
    Xpred (d,n) = w_o + w_item(d)+ w_user(n) with parameters minimizing RMSE.

### Matrix Factorization
- Matrix factorization with Stochastic Gradient Descent (SGD)
introducing ridge terms
tuning parameters by grid search
- Matrix factorization with Alternating Least Squares (ALS)
tuning parameters by grid search
- Predictions with tuned SGD and ALS
- Substraction of baseline before Matrix factorization
Various baseline trial and error tuning
- Parameter tuning by gradient descent in parameter space for SGD, SGD baseline corrected.
Test of tuned parameters with ALS.
- SGD with non linearity
Matrix factorization followed by sigmoid function (taking values between 1 and 5). Modification of learning step.

### Model comparison and variance estimation
- Cross-validation of the different models

### Final prediction
- Final prediction with baseline corrected SGD and tuned parameters.


## How to run the code 

In order to be able to run the code, you should be able to run Jupyter Notebooks (requires the installation of Anaconda), and the following libraries should be installed: numpy and matplotlib.
Notice that you should first download the data to be able to run the code, they are available at AICrowd at the EPFL Recommender system 2019 challenge, in the tab 'resources': https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019/dataset_files, and save the downloaded files in the folder 'Datasets' that should be put in the 'recommender_system' folder. Then you should open jupyter notebook and run the file 'main.ipynib' from 'recommender_system/src/' The code runs sequentially, so it is very important to start running the code from the beginning of the file.
Also, the tuning of the hyperparameters is a long process (several hours) so some of the cells are commented and the resulting hyperparameters are put in the next cell. 

### Folder plan

You should put the downoaded files in the folder 'Datasets' to the folder 'recommender_system'. The folders are organized this way :

- recommender_system
   - README.md
   - Datasets
       - data_train.csv
       - sample_submission.csv
   - plots
   - src
       - main.ipynb
       - run.py
   - results_of_lenthy_computations
       - RMSE_test_tuning_gammas.npy
       - RMSE_test_tuning_lambdas.npy
   - utilities
       - baselines.py
       - helpers.py
       - helpers.py
       - implementations.py
       - plots.py
   
The src folder contains all the main notebook where all the listed methods are used. The project to run is in the following path : /recommender_system/src/main.ipynib.
You can find all the matrix factorization methods in the path : /recommender_system/utilities/implementations.py.
The other files in the folder utilities are all the functions we used to modularize the code in the project.
The plots folder contains all the saved plots produced by the main code.


