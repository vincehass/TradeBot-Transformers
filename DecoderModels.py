import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional
from utils import _easy_mlp, _split_series_time_dims, _merge_series_time_dims



class RNNDecoder(nn.Module):
    def __init__(self,
                 dim_features,
                 dim_hidden_features,
                 num_layers,
                 dim_output,
                 bias = True):
        super().__init__()
        self.dim_features = dim_features
        self.dim_hidden_features = dim_hidden_features
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.bias = bias
        self.rnn = nn.RNN(
            input_size=dim_features,
            hidden_size=dim_hidden_features,     
            num_layers=num_layers,       
            batch_first=True,
            bias = bias   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, dim_features)
        )
        self.readout = nn.Linear(dim_hidden_features, dim_output)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(dim_hidden_features * num_layers, dim_output)
        self.distribution_presigma = nn.Linear(dim_hidden_features * num_layers, dim_output)
        self.distribution_sigma = nn.Softplus()

    def forward(self, in_tensor, in_hidden=None):

        output_tensor, hidden_tensor = self.rnn(in_tensor, in_hidden)
        output_signal = self.readout(output_tensor)
        hidden_tensor = hidden_tensor.view(-1, self.dim_hidden_features*self.num_layers)
        pre_sigma = self.distribution_presigma(hidden_tensor)
        mu_hidden = self.distribution_mu(hidden_tensor)
        sigma_hidden = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        
        return output_signal, hidden_tensor, mu_hidden, sigma_hidden

    def loss(self, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function of the decoder.
        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.
        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        
        #If there is intercation between variables then 
        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]
        pred_true_x = true_value[:, ~mask]



        out_tensor, hidden_tensor , mu_hidden, sigma_hidden = self.forward(torch.cat([hist_encoded, pred_encoded], axis = 1))

        dist_log_prob = torch.distributions.normal.Normal(mu_hidden, sigma_hidden).log_prob(pred_true_x)

        self.dist_extractors = _easy_mlp(
            input_dim=self.dim_features,
            hidden_dim=self.dim_hidden_features,
            output_dim=self.dim_output,
            num_layers=self.num_layers,
            activation=nn.ReLU,
        )
        
        return -dist_log_prob
    
    def sample(
        self, num_samples: int, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]
        device = encoded.device

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        # Ignore the encoding from the historical variables, since there are no interaction between the variables in this decoder.
        pred_encoded = encoded[:, ~mask, :]
        # Except what is needed to copy to the output
        hist_true_x = true_value[:, mask]

        out_tensor, hidden_tensor , mu_hidden, sigma_hidden = self.forward(pred_encoded)
        
        dist = torch.distributions.normal.Normal(mu_hidden, sigma_hidden)
   
        pred_samples = dist.rsample((num_samples,)).permute((1,2, 0))
   
        samples = torch.zeros(num_batches, num_series * num_timesteps, num_samples, device=device)
       
        samples[:, mask, :] = hist_true_x[:, :, None]
       
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, torch.Size((num_batches, num_series, num_timesteps, num_samples)))
    






