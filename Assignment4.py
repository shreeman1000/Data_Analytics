import os
import pickle
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')

class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [10, 20, 40, 70, 190, 110, 150, 180, 200, 250]
        self.L = 10

    def get_predictions(self, X, Z_0 = None, w = 10, L = None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        score = Z_0 * (1 - np.exp(-L * X / Z_0))
        return score

    def loss(self, y_pred, y_actual):
        loss = (y_pred + 1) * np.log((y_pred + 1) / (y_actual + 1)) - y_pred + y_actual
        return loss

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters
                   over the given datapoints.
        """

        Z_0, L = Params

        y_pred = self.get_predictions(X = X, Z_0=Z_0, w = w, L = L)
        loss = self.loss(y_pred, Y)

        return np.mean(loss)



    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)

    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path = '04_cricket_1999to2011.csv') -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    df = pd.read_csv(data_path)
    return df


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    df = data.dropna()
    df = df[df['Error.In.Data'] == 0]
    df['Over'] = 50 - df['Over']

    columns = ['Match', 'Innings', 'Runs.Remaining', 'Wickets.in.Hand','Over', 'Innings.Total.Runs']
    df = df[columns]

    df_1_inning = df[df['Innings'] == 1]

    return df_1_inning

def mean_run_by_wicket(data1, wicket):
    """Helper function for get_Z0"""
    data = data1[data1['Wickets.in.Hand'] == wicket]
    data = data.groupby(['Match'])['Runs.Remaining'].max()

    return data.mean()

def get_Z0(df):
    """get initial estimate of Z0"""
    Z0 = []
    for w in np.arange(10):
        Z0.append(mean_run_by_wicket(df, w+1))

    return Z0

def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model

    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    params = model.Z0 + [model.L]
    print("Z0 =", model.Z0, "L = ", model.L)
    return params

def plot10(Params, plot_path = None) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    Z0_values, L = Params

    overs = np.linspace(0, 50, 300)

    plt.figure(figsize=(10, 8))

    for w, Z0_w in enumerate(Z0_values):
        Z = Z0_w * (1 - np.exp(-(L[w]* overs /Z0_w)))
        plt.plot(overs, Z, label=f'W = {w + 1}, Z = {Z0_w:.1f}, L = {L[w]:.1f}')

    plt.xlabel('Overs')
    plt.ylabel('Z')
    plt.title('Duckworth-Lewis Method Initial 10 runs')
    plt.legend()
    plt.grid()
    plt.xlim(0, 50)
    plt.ylim(0, max(Z0_values))

    if plot_path is not None:
        plt.savefig(plot_path)

    plt.show()

def plot(model: DLModel, plot_path = None) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    Z0_values = model.Z0
    L = model.L

    overs = np.linspace(0, 50, 300)

    plt.figure(figsize=(10, 8))

    for w, Z0_w in enumerate(Z0_values):
        Z = Z0_w * (1 - np.exp(-(L* overs /Z0_w)))
        plt.plot(overs, Z, label=f'W = {w + 1}, Z0 = {Z0_w:.2f}')

    plt.xlabel('Overs')
    plt.ylabel('Z')
    plt.title('Duckworth-Lewis Method')
    plt.legend()
    plt.grid()
    plt.xlim(0, 50)
    plt.ylim(0, max(Z0_values))

    if plot_path is not None:
        plt.savefig(plot_path)

    plt.show()

def calculate_loss(Params, model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on

    Returns:
        float: loss for the given model and data
    '''
    if Params is not None:
        Z_0 = Params[:-1]
        L = Params[-1]
    else:
        Z_0 = model.Z0
        L = model.L

    loss = 0.0

    for wicket in range(1, 11):
        data_temp = data[(data['Wickets.in.Hand'] == wicket)]
        runs, overs = data_temp['Runs.Remaining'].values, data_temp['Over'].values
        loss += model.calculate_loss([Z_0[wicket - 1], L], X = overs, Y = runs, w = wicket)

    return loss

def calculate_loss_wicket(Params, model: DLModel, data: Union[pd.DataFrame, np.ndarray], wicket: int) -> float:
    '''
    Calculates the loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on

    Returns:
        float: loss for the given model and data
    '''
    if Params is not None:
        Z_0 = Params[:-1]
        L = Params[-1]
    else:
        Z_0 = model.Z0
        L = model.L

    data_temp = data[(data['Wickets.in.Hand'] == wicket)]
    runs, overs = data_temp['Runs.Remaining'].values, data_temp['Over'].values
    loss = model.calculate_loss([Z_0, L], X = overs, Y = runs, w = wicket)

    return loss

def get_weighted_avg(data):
    examples = []
    for i in range(1, 11):
        a = data[(data['Wickets.in.Hand'] == i)].shape[0]
        examples.append(a)
    weighted_avg =  [i / sum(examples) for i in examples]
    return weighted_avg


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    Z0 = get_Z0(data)
    L = [5]*10
    losses = []
    #the initial 10 runs to initialise L
    for wicket in range(1, 11):
        Z0_w = Z0[wicket - 1]
        L_w = L[wicket - 1]
        ans = minimize(calculate_loss_wicket, (Z0_w, L_w), args = (model, data, wicket), method='L-BFGS-B')
        Z0[wicket - 1], L[wicket - 1] = ans.x
        losses.append(calculate_loss_wicket(ans.x, model, data, wicket))
    prelim_z = Z0
    prelim_l = L

    avg = get_weighted_avg(data)

    L_final = 0
    for i,j in zip(avg, L):
        L_final += i*j

    Params = Z0 + [L_final]
    ans = minimize(calculate_loss, Params, args = (model, data), method='L-BFGS-B')
    model.Z0 = ans.x[:-1]
    model.L = ans.x[-1]
    loss = calculate_loss(ans.x, model, data)

    return prelim_z, prelim_l, losses, loss

def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")

    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")

    model = DLModel()  # Initializing the model
    prelim_z, prelim_l, losses, loss = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    plot10((prelim_z, prelim_l), args['plot_path2'])
    plot(model, args['plot_path'])  # Plotting the model
    print_model_params(model)
    print("loss =",loss)
    print(prelim_z, prelim_l)
    print("Losses =", losses)

if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot1.png",  # ensure that the path exists
        "plot_path2": "../plots/plot2.png",
    }
    main(args)

