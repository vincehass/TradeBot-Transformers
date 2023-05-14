import torch
import torch.nn as nn
from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import List
import pandas as pd
from datetime import datetime
data = pd.read_csv("Question2.csv", index_col=0, header=[0,1], parse_dates=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        pass

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.
        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.
        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.
    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.
    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=1e-8)
        self.mean = mean

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.
        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.
        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    assert x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


def _easy_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: Type[nn.Module]
) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """
    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)

def plot_single_series(samples, target, timesteps, index):
    s_samples = samples[0, index, :, :].detach().cpu().numpy()
    s_timesteps = timesteps[0, :].cpu().numpy()
    s_target = target[0, index, :].cpu().numpy()
    
    plt.figure()
    
    for zorder, quant, color, label in [
        [1, 0.05, (0.75,0.75,1), "5%-95%"],
        [2, 0.10, (0.25,0.25,1), "10%-90%"],
        [3, 0.25, (0,0,0.75), "25%-75%"],
    ]:
        plt.fill_between(
            s_timesteps,
            np.quantile(s_samples, quant, axis=1),
            np.quantile(s_samples, 1 - quant, axis=1),
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )
    
    plt.plot(
        s_timesteps,
        np.quantile(s_samples, 0.5, axis=1),
        color=(0.5,0.5,0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )
    
    plt.plot(s_timesteps, s_target, color=(0, 0, 0), linewidth=2, zorder=5, label="ground truth")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    
    plt.show()

def hourly_results(v_plus, bid_price, v_neg, offer_price, da, rt):
    
    return (v_plus * (da <= bid_price) * (rt - da)) + (v_neg * (offer_price < da) * (da - rt))


def worst_loss(results):
    
    return min(results.sum(axis=1))




class TimeseriesDataset():
    """
    A Dataset for a multivariate time series.
    It will split the data into historical and prediction values,
    and assign time values to each time step.
    """

    def __init__(self, data):
        """
        Parameters:
        -----------
        data: List[np.array]
            The data for the dataset, as a list of aligned 1d series.
        hist_length: int
            When doing the forecast, how many time steps will be available.
        pred_length: int
            When doing the forecast, the length of said forecast.
        """
        self.data = data

        self.da = self.data["da"]
        self.rt = self.data["rt"]
        self.X = self.data["X"]

        # example of prices with a two day lag if you wish to use timeseries as features (ie RNN, CNN, ARIMA, etc...)
        self.shifted_da = self.da.shift(freq="48H")
        self.shifted_rt = self.rt.shift(freq="48H")

        self.split = datetime(2020,8,1)
        self.X_train = self.X.loc[:self.split]
        self.X_validate = self.X.loc[self.split:]
        self.da_train = self.da.loc[:self.split]
        self.da_validate = self.da.loc[self.split:]
        self.rt_train = self.rt.loc[self.split:]
        self.rt_validate = self.rt.loc[:self.split]


    def features_train(self):
        self.X_train = self.X.loc[:self.split]
        return self.X_train
        
    def features_validate(self):
        self.X_validate = self.X.loc[self.split:]
        return self.X_validate
        
    def dahead_train(self):
        self.da_train = self.da.loc[:self.split]
        return self.da_train
    
    def dahead_validate(self):
        self.da_validate = self.da.loc[self.split:]
        return self.da_train
              
    def realt_train(self):
        self.rt_train = self.rt.loc[self.split:]
        return self.rt_train    

    def realt_validate(self):
        self.rt_validate = self.rt.loc[:self.split]
        return self.rt_train  

# data_ts = TimeseriesDataset(data)