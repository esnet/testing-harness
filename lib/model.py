'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
**
** Date Created:  June 17' 2021
** Last Modified: July 29' 2021
**
'''

from __future__ import absolute_import, print_function

# --- System ---
import os
import sys
import time
import errno
import warnings
warnings.filterwarnings('ignore')

# --- Utility ---
import pandas as pd
import numpy as np
import math
import random
import logging
import pickle
import argparse
from tqdm import tqdm
from datetime import datetime

# --- Pytorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Found {device} available.")
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
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

        # Dropping columns that are not required at the moment
        df = df.drop(columns=['Unnamed: 0','UUID','HOSTNAME','TIMESTAMP',
                              'THROUGHPUT (Receiver)','LATENCY (mean)',
                              'CONGESTION (Receiver)','BYTES (Receiver)'])
        return df

    def _preprocessing(self, df, verbose=False):
        print("\nStarted preprocessing ...")
        # Spliting 1gbps -> 1, gbps
        pacing = df['PACING'].values
        for i, p in enumerate(pacing):
            v, _ = p.split("gbit")
            pacing[i] = float(v)
        df['PACING'] = pacing

        # Dropping rows with pacing rate 10.5, glitch in the training data
        df.drop( df[ df['PACING'] == 10.5 ].index, inplace=True)
        # Supervised training approach needs total number of classes for classification task
        num_of_classes = len(df['PACING'].unique())

        if verbose:
            print(f"Using the following features:\n{clr.G}{df.columns.values}{clr.E}\n")

        """
        Transform between iterable of iterables and a multilabel format.
        Although a list of sets or tuples is a very intuitive format for
        multilabel data, it is unwieldy to process. This transformer converts
        between this intuitive format and the supported multilabel format
        """
        mlb = MultiLabelBinarizer(sparse_output=True)
        alias_df = df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df.pop('ALIAS')),
                                                            index=df.index,
                                                            columns=mlb.classes_))

        df_ = alias_df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(alias_df.pop('CONGESTION (Sender)')),
                                                             index=alias_df.index,
                                                             columns=mlb.classes_),
                            how = 'left', lsuffix='left', rsuffix='right')

        y = df_['PACING'].values
        y = y.astype('float')

        df_train = df_.drop(['PACING'], axis=1)
        X = df_train[df_train.columns.values].values

        """
        Normalization: This estimator scales and translates each feature individually 
        such that it is in the given range on the training set
        """
        minmax_scale = preprocessing.MinMaxScaler().fit(df_train[df_train.columns.values])
        df_minmax = minmax_scale.transform(df_train[df_train.columns.values])

        final_df = pd.DataFrame(df_minmax, columns=df_train.columns.values)
        X = final_df[df_train.columns.values].values

        return X, y, num_of_classes


# Custom data loader for ELK stack dataset
class PACINGDATASET(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# model definition
class PACINGCLASSIFIER (nn.Module):
    """
    Pacing Classifier is a supervised approach to the pacing
    prediction task (assuming interface limit 10G)
    """
    def __init__(self, nc=20, inputFeatures=7):
        super(PACINGCLASSIFIER, self).__init__()

        self.fc1 = torch.nn.Linear(inputFeatures, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64, 128)
        self.fc4 = torch.nn.Linear(128, 256)
        self.fc5 = torch.nn.Linear(256, 128)
        self.fc6 = torch.nn.Linear(128, 64)
        self.fc7 = torch.nn.Linear(64, nc)

        """
        Fills the input Tensor with values according to the method
        described in "Understanding the difficulty of training deep
        feedforward neural networks" - Glorot, X. & Bengio, Y. (2010),
        using a uniform distribution. The resulting tensor will have
        values sampled from mathcal{U}(-a, a)U(−a,a)
        """
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.zeros_(self.fc4.bias)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.zeros_(self.fc5.bias)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.zeros_(self.fc6.bias)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.025)

    def forward(self, x):
        z = self.lrelu(self.fc1(x))
        z = self.lrelu(self.fc2(z))
        z = self.lrelu(self.fc3(z))
        z = self.lrelu(self.fc4(z))
        z = self.lrelu(self.fc5(z))
        z = self.lrelu(self.fc6(z))
        z = self.fc7(z)  # no activation
        return z

    def _train(self, args, model, trainloader, testloader, optimizer, scheduler, lossFunction):
        # Model training on the retrieved statistics.
        print("")
        print("Epoch", " Loss", "  Acc", sep=' '*11, end="\n")
        EPOCH = args.epoch
        trainloss = []
        
        for epoch in range(0, EPOCH):
            model.train()
            torch.manual_seed(epoch+1)                      # recovery reproducibility
            epoch_loss = 0                                  # for one full epoch

            for (batch_idx, batch) in enumerate(trainloader):
                (xs, ys) = batch                            # (predictors, targets)
                xs, ys = xs.float(), ys.float()
                optimizer.zero_grad()                       # prepare gradients

                output = model(xs)                          # predicted pacing rate
                loss = lossFunction(output, ys.long())      # avg per item in batch

                epoch_loss += loss.item()                   # accumulate averages
                loss.backward()                             # compute gradients
                optimizer.step()                            # update weights
            
            # scheduler.step()
            trainloss.append(epoch_loss)
            if epoch % args.interval == 0:

                model.eval()                                # evaluation phase on every epoch
                correct, acc = 0, 0
                with torch.no_grad():
                    for xs, ys in testloader:
                        xs, ys = xs.float(), ys.long()
                        pred = torch.max(model(xs), 1)[1]
                        correct += (pred == ys).sum().item()
                    acc = (100 * float(correct / len(testloader.dataset)) )

                print(f"{epoch+0:03}/{EPOCH}", f"{epoch_loss:.4f}", f"{acc:.4f}", sep=' '*10, end="\n")

                dt = time.strftime("%Y_%m_%d-%H_%M_%S")
                fn = "checkpoint/" + str(dt) + str("-") + str(epoch) + "_ckpt.pt"

                info_dict = {
                    'epoch' : epoch,
                    'model_state' : model.state_dict(),
                    'optimizer_state' : optimizer.state_dict()
                }
                if args.save:
                    torch.save(info_dict, fn)               # save checkpoint

        torch.save (info_dict, "checkpoint/best.pt")
        print("*"*25)
        print("Training complete\n")
        return model

    def _loadModel(self, fn, num_of_classes, inputFea, verbose=False):
        # Load a pre-trained model from a given path
        model = PACINGCLASSIFIER (nc=num_of_classes, inputFeatures=inputFea)
        modelPath = torch.load(fn)
        model.load_state_dict(modelPath['model_state'])
        model.to(device)
        print("*"*25)
        print("Pre-trained model loaded")
        print("*"*25)
        return model

    def _test(self, model, inputSample, inputFea):

        if len(inputSample)==inputFea and isinstance(inputSample, list):
            # converting the sample to tensor array
            ukn = np.array([inputSample], dtype=np.float32)
            sample = torch.tensor(ukn, dtype=torch.float32).to(device)

        elif len(inputSample)!=inputFea and isinstance(inputSample, list):
            # Needs pre-processing similar to model training

            # converting the sample to tensor array
            ukn = np.array([inputSample], dtype=np.float32)
            sample = torch.tensor(ukn, dtype=torch.float32).to(device)
        
        elif isinstance(inputSample, torch.Tensor):
            # Using a test data sample for Demo
            sample = inputSample.float().unsqueeze_(0)


        # Inference stage
        model.eval()
        with torch.no_grad():
            pred = torch.max(model(sample), 1)[1]
        return pred.item()


def getPacingRate(bufferData, phase='test', verbose=False):

    seeder = SEEDEVERYTHING()
    seeder._weight_init_()

    # Preprocessing
    prep = DATA("data/statistics-5.csv")
    df = prep._df_load_and_clean("data/statistics-5.csv")

    if bufferData:
        df = df.append({
                        'ALIAS':"hostA", # bufferData[0],
                        'STREAMS':bufferData[1],
                        'PACING':"6gbit",
                        'THROUGHPUT (Sender)':bufferData[2],
                        'LATENCY (min.)':bufferData[3],
                        'LATENCY (max.)':bufferData[4],
                        'RETRANSMITS':bufferData[5],
                        'CONGESTION (Sender)':bufferData[6],
                        }, ignore_index=True)

    X, y, num_of_classes = prep._preprocessing(df)
    X = torch.tensor(X)
    y = torch.tensor(y)

    # Dataset w/o any tranformations
    data = PACINGDATASET(tensors=(X, y), transform=None)
    dataloader  = torch.utils.data.DataLoader(data, batch_size=256)

    inputFea = len(data[0][0])
    if verbose:
        print("Length of Input Feautures: ", inputFea)
        print("Length of Input: ", len(bufferData))

    model = PACINGCLASSIFIER (nc=num_of_classes, inputFeatures=inputFea)
    if verbose:
        print("\n", model)

    fn = os.path.join(os.getcwd(), "checkpoint/best.pt")
    if verbose:
        print(f"Current working directory: {fn}")

    if phase=="test" and os.path.exists(fn):
        try:
            # Get the features from iperf3 prob test

            print("\nInside the inference stage")
            # Load the model
            inferenceModel = model._loadModel(fn, num_of_classes, inputFea)
            if verbose:
                print(f"printing the input sample: {data[len(data)-1]}\n")
            inputSample, groundtruth = data[len(data)-1]

            pacing = model._test(inferenceModel, inputSample, inputFea)
            print(f"Predicted pacing rate: {clr.G}{pacing}{clr.E}")

        except Exception as e:
            print(f"Exception error: {e}")
            print(f"{clr.F}Note: checkpoint folder should contain a pre-trained model, or switch the training phase.{clr.E}")

    return pacing


def main():

    parser = argparse.ArgumentParser(description='Testpoint Statistics')

    parser.add_argument('-p', '--phase', default="test", type=str,
                        help='Training and Testing Phase. {train/test}')
    parser.add_argument('--infile', default="data/statistics-5.csv", type=str,
                        help='CSV file used for training the model.')

    parser.add_argument('-e', '--epoch', default=300, type=int,
                        help='Total number of training epochs')
    parser.add_argument('-b', '--batch', default=256, type=int,
                        help='Batch-size in dataloader')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                        help='Learning rate for the optimizer')
    parser.add_argument('-i', '--interval', default=25, type=int,
                        help='Print statement interval')
    parser.add_argument('-s', '--save', action='store_true',
                        help='To save the checkpoints of the training model')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='To save the checkpoints of the training model')

    args = parser.parse_args()
    print("")
    for arg in vars(args):
        print ("%-15s: %s"%(arg,getattr(args, arg)))

    BESTLOSS = 10

    print("")
    seeder = SEEDEVERYTHING()
    seeder._weight_init_()

    # Preprocessing
    prep = DATA(args.infile)
    df = prep._df_load_and_clean(args.infile)
    X, y, num_of_classes = prep._preprocessing(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.25,
                                                random_state=1)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test)

    lossFunction  = nn.CrossEntropyLoss()
    # BCE = nn.BCELoss(reduction='mean')
    # MSE = nn.MSELoss(reduction='mean') # 'mean', 'sum'. 'none'

    # Dataset w/o any tranformations
    traindata   = PACINGDATASET(tensors=(X_train, y_train),
                                transform=None)
    trainloader = torch.utils.data.DataLoader(traindata,
                                              shuffle=True,
                                              batch_size=args.batch)

    testdata    = PACINGDATASET(tensors=(X_test, y_test),
                                transform=None)
    testloader  = torch.utils.data.DataLoader(testdata,
                                              shuffle=True,
                                              batch_size=args.batch)

    print()
    inputFea = len(traindata[0][0])
    model = PACINGCLASSIFIER (nc=num_of_classes, inputFeatures=inputFea)
    print("\n", model)

    optimizer = optim.SGD(model.parameters(),
                         lr=args.learning_rate,
                         momentum=0.9,
                         # weight_decay=5e-4,
                         # nesterov=True,
                         )
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=args.learning_rate,
    #                        weight_decay=5e-4,
    #                        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[35],
                                                    gamma=0.1)

    fn = os.path.join(rootdir,"checkpoint/best.pt")
    if args.phase=="test" and os.path.exists(fn):
        try:
            # Get the features from iperf3 prob test
            inferenceModel = model._loadModel(fn, num_of_classes, inputFea)

            inputSample, groundtruth = testdata[100]

            pacing = model._test(inferenceModel, inputSample, inputFea)
            print(f"Normalized Input Sample :\n{clr.G}{inputSample}{clr.E}")
            print(f"Groundtruth pacing rate: {clr.G}{groundtruth.item()}{clr.E}\nPredicted pacing rate: {clr.G}{pacing}{clr.E}\n")
        except:
            # DO THE TRAINING
            ckpt = model._train(args, model, trainloader, testloader, optimizer, scheduler, lossFunction)          

    elif args.phase=="train":
        # DO THE TRAINING
        ckpt = model._train(args, model, trainloader, testloader, optimizer, scheduler, lossFunction)

        inferenceModel = model._loadModel(fn, num_of_classes, inputFea)
        
        inputSample, groundtruth = testdata[101]
        pacing = model._test(inferenceModel, inputSample, inputFea)
        print(f"Normalized Input Sample :\n{clr.G}{inputSample}{clr.E}")
        print(f"Groundtruth pacing rate: {clr.G}{groundtruth.item()}{clr.E}\nPredicted pacing rate:  {clr.G}{pacing}{clr.E}\n")


if __name__ == "__main__":
    main()