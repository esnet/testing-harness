'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
**
** Date Created:  June 17' 2021
** Last Modified: Aug 20' 2021
**
'''

from __future__ import absolute_import, print_function

# --- System ---
import os
import sys
import errno
import warnings
warnings.filterwarnings('ignore')

# --- Utility ---
import pandas as pd
import numpy as np
import math
import random
import pickle
import argparse

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# --- Sklearn ---
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition, discriminant_analysis, linear_model, tree, neural_network

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


rootdir = os.getcwd()

try:
    os.makedirs(os.path.join(rootdir,'checkpoint'))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


class clr:
    """
    Defining colors for the print syntax coloring
    """
    H   = '\033[35m' # Header
    B   = '\033[94m' # Blue
    G   = '\033[36m' # Green
    W   = '\033[93m' # Warning
    F   = '\033[91m' # Fail
    E   = '\033[0m'  # End
    BD  = '\033[1m'  # Bold
    UL  = '\033[4m'  # Underline


class SEEDEVERYTHING:
    # random weight initialization
    def __init__(self):
        self.seed = 42

    def _weight_init_(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        print("Seeded everything.")


class DATA:
    """
    Providing all the preprocessing techniques such as;
    - Data Loading,
    - Data Cleaning,
    - One-Hot Encoding,
    - Data Normalization
    """
    def __init__(self, infile):
        self.infile = infile

    def _df_load_and_clean(self, infile):
        # data loading and preprocessing
        dataPath = self.infile
        df = pd.read_csv(dataPath)

        # Renaming the column names for easier access
        df.columns = ['Unnamed: 0', 'UUID', 'HOSTNAME', 'ALIAS', 'TIMESTAMP', 'STREAMS',
                      'PACING', 'THROUGHPUT-S', 'THROUGHPUT-R', 'LATENCY-min', 'LATENCY-max', 
                      'LATENCY-mean', 'RETRANSMITS', 'CONGESTION-S', 'CONGESTION-R', 'BYTES-R']

        # Dropping columns that are not required at the moment
        df = df.drop(columns=['Unnamed: 0', 'UUID', 'HOSTNAME', 'TIMESTAMP', 'CONGESTION-R'])
        return df

    def _compositeOfIntersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value not in lst2]
        return lst3

    def _preprocessing(self, df, verbose=False):
        print("Started preprocessing ...")
        
        """
        Step 1. spliting 1gbps -> 1, gbps
        Step 2. dropping rows with pacing rate 10.5, glitch in the training data
        Step 3. applying labelEncoder to encode the labels
        Step 4. applying transformation to the encoded labels after 
                prediction in the final model get the decoded label 
                by: "le.inverse_transform([])"
        Step 5. drop the pacing from the intput (X) feautures
        """
        pacing = df['PACING'].values # (1)
        for i, p in enumerate(pacing):
            v, _ = p.split("gbit")
            pacing[i] = float(v)
        df['PACING'] = pacing
        df.drop( df[ df['PACING'] == 10.5 ].index, inplace=True) # (2)

        y = df['PACING'].values
        le = preprocessing.LabelEncoder() # (3)
        le.fit(y)
        y = le.transform(y) # (4)
        df = df.drop(['PACING'], axis=1) # (5)
        # y = y.astype('float64')

        if verbose:
            print(f"Using the following features:\n{clr.G}{df.columns.values}{clr.E}\n")

        """
        Applying get dummies to all the categorical columns to 
        increase the number of columns.
        ['ALIAS'] = {'hostA', 'hostB', ...}
        ['STREAMS'] = {1, 4, 8, 16}
        ['CONGESTION-S'] = {'bbr2', 'cubic'}
        """
        df_alias = pd.get_dummies(df['ALIAS'])
        df_streams = pd.get_dummies(df['STREAMS'])
        df_congestion = pd.get_dummies(df['CONGESTION-S'])
        df2 = pd.concat([df, df_alias, df_streams, df_congestion], axis=1)
        df2 = df2.drop(['ALIAS', 'CONGESTION-S'], axis=1)
        df2['S1'],df2['S4'],df2['S8'],df2['S16'] = df2[1],df2[4],df2[8],df2[16]
        df3 = df2.drop([1,4,8,16], axis=1)

        """
        Normalization: This estimator scales and translates each feature individually 
        such that it is in the given range on the training set
        """
        totalColumnList = df3.columns
        columnListNorm = ['THROUGHPUT-S', 'THROUGHPUT-R',  'LATENCY-min', 'LATENCY-max', 'LATENCY-mean', 'RETRANSMITS', 'BYTES-R']
        remainingColumnList = self._compositeOfIntersection(totalColumnList, columnListNorm)
        df_minmax = preprocessing.StandardScaler().fit_transform(df3[columnListNorm])
        df4 = pd.DataFrame(df_minmax, columns=columnListNorm)

        df3.reset_index(drop=True, inplace=True)
        df4.reset_index(drop=True, inplace=True)
        df5 = pd.concat([df3[remainingColumnList], df4[columnListNorm]], axis=1)
        df5 = df5.astype('float64')

        # Converting dataframe to nd.array 
        X_ = df5[df5.columns.values].values
        X = X_.astype('float64')
        num_of_classes = len(np.unique(y))

        if verbose:
            print(df5.head())

        return X, y, num_of_classes, le


# model definition
class PACINGCLASSIFIER:
    def __init__(self, modelName):
        self.modelName = modelName

    def _defineModel (self):
        if self.modelName=='rf':
            model = RandomForestClassifier(random_state=999)
            params = {
                "n_estimators": [100],
                "criterion": ["gini","entropy"],
                "max_depth": [2,4,6,8],
                "min_samples_leaf": [1,2,5,10,15,20,25,48],
            }

        elif self.modelName=='dtr':
            model = tree.DecisionTreeRegressor(random_state=999)
            params = {
                "criterion": ["mse", "friedman_mse", "mae", "poisson"],
                "max_depth": [8,12,16],
                "min_samples_split": [2,4,8,12,24,48],
                "splitter": ["best", "random"],
            }

        return model, params

    def _trainAndTune (self, X, y, model, parameters, kfcv, verbose):
        # Fine-tuning hyperparameters of the model using grid cv
        cvSearchObj = GridSearchCV(model, parameters, scoring='f1_macro', n_jobs=-1, cv=kfcv, verbose=verbose)
        cvSearchObj.fit(X,y)
        if verbose:
            print("Best score: ", cvSearchObj.best_score_)
        return cvSearchObj.best_estimator_, cvSearchObj.best_params_


    def _saveModel (self, fn, model):
        # Saving the trained model given filename
        pickle.dump(model, open(fn, 'wb'))
        print(f"Model saved as {fn}")


    def _loadModel (self, filename, verbose=False):
        # Load a pre-trained model from a given path
        model_reloaded = pickle.load(open(filename, 'rb'))
        print("Pre-trained model loaded")
        return model_reloaded


    def _train (self, X_train, y_train, model, params, kfcv, verbose=False):
        # Model training on the retrieved statistics
        modelBest, bestparams = self._trainAndTune(X_train,
                                                   y_train, 
                                                   model, 
                                                   params, 
                                                   kfcv,
                                                   verbose)

        if verbose:
            print("Best hyperparameters: ", bestparams)

        print("Training complete\n")
        return modelBest


    def _test (self, modelReloaded, inputSample, inputFeatureSize, le):
        output = None
        # Model testing on the retrieved statistics
        if len(inputSample)==23 or len(inputSample)==24:
            pred = modelReloaded.predict(inputSample.reshape((1,inputFeatureSize)))
            output = le.inverse_transform(pred)
        else:
            pred = modelReloaded.predict(X_test)
            acc = modelReloaded.score(X_test, y_test)
            print("Testing accuracy: {acc:.2f}%")
        return output.item()


def getPacingRate(bufferData, phase='test', verbose=False):

    seeder = SEEDEVERYTHING()
    seeder._weight_init_()

    # Preprocessing
    prep = DATA("data/statistics-5.csv")
    df = prep._df_load_and_clean("data/statistics-5.csv")

    if bufferData:
        df = df.append({
                        'ALIAS':bufferData[0],
                        'STREAMS':bufferData[1],
                        # This is just to replace N.A/missing value during 
                        # appending it in the dataframe, eventually we'll predict this value
                        'PACING':"5gbit",
                        'THROUGHPUT-S':bufferData[2],
                        'THROUGHPUT-R':bufferData[3],
                        'LATENCY-min':bufferData[4],
                        'LATENCY-max':bufferData[5],
                        'LATENCY-mean':bufferData[6],
                        'RETRANSMITS':bufferData[7],
                        'CONGESTION-S':bufferData[8],
                        'BYTES-R':bufferData[9],
                        }, ignore_index=True)

    X, y, num_of_classes, le = prep._preprocessing(df)

    pacingclassifier = PACINGCLASSIFIER(modelName='rf')
    model, params = pacingclassifier._defineModel()

    fn = "checkpoint/rfBest.pkl"
    # Reload the model
    model_reloaded = pacingclassifier._loadModel (fn)
    # Apply testing sample/data
    pacing = pacingclassifier._test(model_reloaded, X[len(X)-1], len(X[len(X)-1]), le)
    print(f"Predicted pacing rate: {clr.H}{pacing}{clr.E}")

    return pacing


def main():

    parser = argparse.ArgumentParser(description='Testpoint Statistics')

    parser.add_argument('-p', '--phase', default="test", type=str,
                        help='Training and Testing Phase. {train/test}')
    
    parser.add_argument('--infile', default="data/statistics-5.csv", type=str,
                        help='CSV file used for training the model.')

    parser.add_argument('-k', '--kv', default=5, type=int,
                        help='K-fold Cross Validation')

    parser.add_argument('--split', default=25, type=int,
                        help='Dataset split percentage (10, 25, 33)')

    parser.add_argument('-s', '--save', action='store_true',
                        help='To save the checkpoints of the training model')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='v flag prints all the steps results throughout the process')
    
    parser.add_argument('--modelName', default="rf", type=str,
                        help="Optimizer {rf- Random Forest / dtr- decision tree regressor }")

    args = parser.parse_args()
    print("")
    for arg in vars(args):
        print ("%-15s: %s"%(arg,getattr(args, arg)))

    print("")
    seeder = SEEDEVERYTHING()
    seeder._weight_init_()

    # Preprocessing
    prep = DATA(args.infile)
    df = prep._df_load_and_clean(args.infile)
    X, y, num_of_classes, le = prep._preprocessing(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.33,
                                                random_state=1)
    
    pacingclassifier = PACINGCLASSIFIER(modelName=args.modelName)
    model, params = pacingclassifier._defineModel()

    if args.phase=="train":
        trainedPacingModel = pacingclassifier._train(X_train, y_train, model, params, args.kv)
        fn = "checkpoint/rfBest.pkl"
        # Save the model
        pacingclassifier._saveModel (fn, trainedPacingModel)
        # Reload the model
        model_reloaded = pacingclassifier._loadModel (fn)
        # Apply testing sample/data
        pacingRate = pacingclassifier._test(model_reloaded, X_test[0], len(X_test[0]), le)
        print("Predicted pacing rate: ", pacingRate)
    
    elif args.phase=="test":
        fn = "checkpoint/rfBest.pkl"
        # Reload the model
        model_reloaded = pacingclassifier._loadModel (fn)
        # Apply testing sample/data
        pacingRate = pacingclassifier._test(model_reloaded, X_test[0], len(X_test[0]), le)
        print(f"Predicted pacing rate: {clr.H}{pacingRate}{clr.E}")

if __name__ == "__main__":
    main()