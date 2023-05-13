import torch
import torch.nn as nn
import numpy as np
from utils import _easy_mlp, _merge_series_time_dims, _split_series_time_dims, NormalizationIdentity, NormalizationStandardization
from typing import Optional, Dict, Any
from utils import device
import pandas as pd
import cvxpy as cp



class TradingBot(nn.Module):

    """
    The top-level module for TradingBot.

    The role of this module is to handle everything outside of the encoder and decoder Model.
    This consists mainly the data manipulation ahead of the encoder and after the decoder.
    """

    def __init__(
        self,
        num_series: int,
        input_dim :int,
        series_embedding_dim: int,
        input_encoder_layers: int,
        gamma: float ,
        l_norm: int,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        #positional_encoding: Optional[Dict[str, Any]] = None,
        encoder: Optional[Dict[str, Any]] = None,
        temporal_encoder: Optional[Dict[str, Any]] = None,
        rnn_decoder: Optional[Dict[str, Any]] = None,
        gaussian_decoder: Optional[Dict[str, Any]] = None,
        percentile: float = 0.05,
        max_loss: float = -1000.0,
        lookback_window : int = 24,
        lookahead_window : int = 24,
       
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        series_embedding_dim: int
            The dimensionality of the per-series embedding.
        input_encoder_layers: int
            Number of layers in the MLP which encodes the input data.
        bagging_size: Optional[int], default to None
            If set, the loss() method will only consider a random subset of the series at each call.
            The number of series kept is the value of this parameter.
        input_encoding_normalization: bool, default to True
            If true, the encoded input values (prior to the positional encoding) are scaled
            by the square root of their dimensionality.
        data_normalization: str ["", "none", "standardization"], default to "series"
            How to normalize the input values before sending them to the model.
        loss_normalization: str ["", "none", "series", "timesteps", "both"], default to "series"
            Scale the loss function by the number of series, timesteps, or both.
        positional_encoding: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a PositionalEncoding for the time encoding.
            The options sent to the PositionalEncoding is content of this dictionary.
        encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a Encoder as the encoder.
            The options sent to the Encoder is content of this dictionary.
        temporal_encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a TemporalEncoder as the encoder.
            The options sent to the TemporalEncoder is content of this dictionary.
        model_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a CopulaDecoder as the decoder.
            The options sent to the CopulaDecoder is content of this dictionary.
        """
        super().__init__()

        self.num_series = num_series
        self.series_embedding_dim = series_embedding_dim
        self.input_encoder_layers = input_encoder_layers
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization
        self.gamma = gamma,
        self.l_norm = l_norm
        self.percentile = percentile
        self.max_loss = max_loss
        
        self.lookback_window = lookback_window,
        self.lookahead_window = lookahead_window,
        
        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
        }[data_normalization]
       
        self.series_encoder = nn.Embedding(num_embeddings=num_series, embedding_dim=self.series_embedding_dim)
        if rnn_decoder is not None:
            self.decoder = RNNDecoder(input_dim, **rnn_decoder)
        if gaussian_decoder is not None:
            self.decoder = GaussianDecoder(input_dim, **gaussian_decoder)
        
    def loss(
        self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor, pred_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function of the model.

        Parameters:
        -----------
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        pred_value: Tensor [batch, series, time steps]
            A tensor containing the values that the model should learn to forecast at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function of TACTiS, with lower values being better. Averaged over batches.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_value.shape[2]
        device = hist_value.device

        # Gets the embedding for each series [batch, series, embedding size]
        # Expand over batches to be compatible with the bagging procedure, which select different series for each batch
        series_emb = self.series_encoder(torch.arange(num_series, device=device))
        series_emb = series_emb[None, :, :].expand(num_batches, -1, -1)

        # Make sure that both time tensors are in the correct format
        if len(hist_time.shape) == 2:
            hist_time = hist_time[:, None, :]
        if len(pred_time.shape) == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, num_series, -1)
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, num_series, -1)


        # The normalizer uses the same parameters for both historical and prediction values
        normalizer = self.data_normalization(hist_value)
        hist_value = normalizer.normalize(hist_value)
        pred_value = normalizer.normalize(pred_value)

        hist_encoded = torch.cat(
            [
                hist_value[:, :, :, None],
                series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                torch.ones(num_batches, num_series, num_hist_timesteps, 1, device=device),
            ],
            dim=3,
        )
        # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
        pred_encoded = torch.cat(
            [
                torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
            ],
            dim=3,
        )

        encoded = torch.cat([hist_encoded, pred_encoded], dim=2)
        #encoded = self.input_encoder(encoded)

        # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
        # Adjustments may be required for other ways to encode time.
        # if self.time_encoding:
        #     timesteps = torch.cat([hist_time, pred_time], dim=2)
        #     encoded = self.time_encoding(encoded, timesteps.to(int))

        # encoded = self.encoder.forward(encoded)

        mask = torch.cat(
            [
                torch.ones(num_batches, num_series, num_hist_timesteps, dtype=bool, device=device),
                torch.zeros(num_batches, num_series, num_pred_timesteps, dtype=bool, device=device),
            ],
            dim=2,
        )
        true_value = torch.cat(
            [
                hist_value,
                pred_value,
            ],
            dim=2,
        )

        loss = self.decoder.loss(encoded, mask, true_value)
        if self.loss_normalization in {"series", "both"}:
            loss = loss / num_series
        if self.loss_normalization in {"timesteps", "both"}:
            loss = loss / num_pred_timesteps
        return loss.mean()
    
    
    
    def train_step(self, optimizer, batch_size, data, hist_length, pred_length):
        max_idx = data.shape[1] - (hist_length + pred_length)
        
        hist_values = []
        pred_values = []
        for _ in range(batch_size):
            idx = np.random.randint(0, max_idx)
            hist_values.append(data[:, idx:idx+hist_length])
            pred_values.append(data[:, idx+hist_length:idx+hist_length+pred_length])
        
        # [batch, series, time steps]
        hist_value = torch.Tensor(hist_values).to(device)
        pred_value = torch.Tensor(pred_values).to(device)
        hist_time = torch.arange(0, hist_length, device=device)[None, :].expand(batch_size, -1)
        pred_time = torch.arange(hist_length, hist_length + pred_length, device=device)[None, :].expand(batch_size, -1)
        
        optimizer.zero_grad()
        loss = self.loss(hist_time, hist_value, pred_time, pred_value)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    

    def predict_samples(self, num_samples, data, hist_length, pred_length):
        max_idx = data.shape[1] - (hist_length + pred_length)

        idx = np.random.randint(0, max_idx)
        hist_value = torch.Tensor(data[:, idx:idx+hist_length]).to(device)
        pred_value = torch.Tensor(data[:, idx+hist_length:idx+hist_length+pred_length]).to(device)

        # [batch, series, time steps]
        hist_value = hist_value[None, :, :]
        pred_value = pred_value[None, :, :]
        hist_time = torch.arange(0, hist_length, device=device)[None, :]
        pred_time = torch.arange(hist_length, hist_length + pred_length, device=device)[None, :]

        samples = self.sample(num_samples, hist_time, hist_value, pred_time)

        return samples, torch.cat([hist_value, pred_value], axis=2), torch.cat([hist_time, pred_time], axis=1)

    
    def sample(
        self, num_samples: int, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the available values
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times at which we want forecasts.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_time.shape[-1]
        device = hist_value.device

        # Gets the embedding for each series [batch, series, embedding size]
        # Expand over batches to be compatible with the bagging procedure, which select different series for each batch
        series_emb = self.series_encoder(torch.arange(num_series, device=device))
        series_emb = series_emb[None, :, :].expand(num_batches, -1, -1)

        # Make sure that both time tensors are in the correct format
        if len(hist_time.shape) == 2:
            hist_time = hist_time[:, None, :]
        if len(pred_time.shape) == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, num_series, -1)
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, num_series, -1)

        # The normalizer remembers its parameter to reverse it with the samples
        normalizer = self.data_normalization(hist_value)
        hist_value = normalizer.normalize(hist_value)

        hist_encoded = torch.cat(
            [
                hist_value[:, :, :, None],
                series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                torch.ones(num_batches, num_series, num_hist_timesteps, 1, device=device),
            ],
            dim=3,
        )
        pred_encoded = torch.cat(
            [
                torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
            ],
            dim=3,
        )

        encoded = torch.cat([hist_encoded, pred_encoded], dim=2)
        #encoded = self.input_encoder(encoded)
        # if self.input_encoding_normalization:
        #     encoded *= self.encoder.embedding_dim**0.5

        # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
        # Adjustments may be required for other ways to encode time.
        # if self.time_encoding:
        #     timesteps = torch.cat([hist_time, pred_time], dim=2)
        #     encoded = self.time_encoding(encoded, timesteps.to(int))

        #encoded = self.encoder.forward(encoded)

        mask = torch.cat(
            [
                torch.ones(num_batches, num_series, num_hist_timesteps, dtype=bool, device=device),
                torch.zeros(num_batches, num_series, num_pred_timesteps, dtype=bool, device=device),
            ],
            dim=2,
        )
        true_value = torch.cat(
            [
                hist_value,
                torch.zeros(num_batches, num_series, num_pred_timesteps, device=device),
            ],
            dim=2,
        )

        samples = self.decoder.sample(num_samples, encoded, mask, true_value)

        samples = normalizer.denormalize(samples)
        return samples

    def predict(self,X, da, da_validate, rt_validate):

        
        samples, pred_value, timesteps = self.predict_samples(X.shape[0], da_validate.values.T, 24,24)#self.lookback_window, self.lookahead_window)
        samples_resh = samples.mean(axis = 2).squeeze(0)
        estimated_market = samples_resh.detach().numpy()
        self.estimated_market_df = pd.DataFrame(estimated_market.T, columns = da_validate.columns)
        
        bid_price = da_validate.mean(axis=0).values-estimated_market.mean()/estimated_market.std()
        offer_price = da_validate.mean(axis=0).values+estimated_market.mean()/estimated_market.std()
        
        v_long, bid, v_short, offer, pb_value = self.maximize_trade_constrain_downside(bid_price, offer_price, da_validate, rt_validate, 
                                                                                     self.percentile, self.max_loss, self.gamma)
                                                                                     
        
        
        return v_long, bid, v_short, offer
    

    def maximize_trade_constrain_downside(self,bid_price, offer_price, da_validate, rt_validate, percentile, max_loss, gamma):

        bid_return = (da_validate <= bid_price) * (rt_validate - da_validate)
        offer_return = (offer_price < da_validate) * (da_validate - rt_validate)                                                  
        

        weights1 = cp.Variable(bid_return.mean(axis=0).shape)
        weights2 = cp.Variable(offer_return.mean(axis=0).shape)
        # print('bid_return.shape',bid_return.shape)
        # print('weights1.shape',weights1.shape)
        # print('offer_return.shape',offer_return.shape)
        # print('weights2.shape',weights2.shape)
        
        objective = cp.Maximize(weights1* bid_return.mean(axis=0)+ weights2* offer_return.mean(axis=0))
        
        

        nsamples = round(bid_return.shape[0]*self.percentile)
        
        portfolio_rets = weights1*bid_return.T + weights2*offer_return.T
        wors_hour = cp.sum_smallest(portfolio_rets, nsamples)/nsamples
        
        
        constraints = [wors_hour>=max_loss, weights1>=0, weights2>=0, cp.norm(weights2, self.l_norm) <= self.gamma,
                                                                        cp.norm(weights1, self.l_norm) <= self.gamma]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return weights1.value.round(4).ravel(), bid_return, weights2.value.round(4).ravel(), offer_return, problem.value



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

        # Ignore the encoding from the historical variables, if you want tio model no interaction between the variables in this decoder.
        # pred_encoded = encoded[:, ~mask, :]
        # pred_true_x = true_value[:, ~mask]
        
        #If there is intercation between variables then 
        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]
        pred_true_x = true_value[:, ~mask]



        # print(' pred_true_x.shape',pred_true_x.shape)
        out_tensor, hidden_tensor , mu_hidden, sigma_hidden = self.forward(torch.cat([hist_encoded, pred_encoded], axis = 1))
        # out_tensor, hidden_tensor , mu_hidden, sigma_hidden = self.forward(pred_encoded)
        # print('pred_encoded.shape',pred_encoded.shape)
        # print('mu_hidden.shape',mu_hidden.shape)
        # print('sigma_hidden.shape',sigma_hidden.shape)
        # print('pred_true_x',pred_true_x.mean())
        dist_log_prob = torch.distributions.normal.Normal(mu_hidden, sigma_hidden).log_prob(pred_true_x)
        #pred = dist.sample()  # not scaled
        #print(pred.shape)
        self.dist_extractors = _easy_mlp(
            input_dim=self.dim_features,
            hidden_dim=self.dim_hidden_features,
            output_dim=self.dim_output,
            num_layers=self.num_layers,
            activation=nn.ReLU,
        )
        #ll = self.dist_extractors(pred)
        #print(ll.shape)
        # logits = self.dist_extractors(pred)#[:, 1:, :]
        # logprob = math.log(self.dim_output) + nn.functional.log_softmax(logits, dim=1)
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
        # print(mu_hidden.shape)
        dist = torch.distributions.normal.Normal(mu_hidden, sigma_hidden)
        # rsamples have the samples as the first dimension, so send it to the last dimension
        # print(dist.rsample((num_samples,)).shape)
        pred_samples = dist.rsample((num_samples,)).permute((1,2, 0))
        # print('pred_samples.shape',pred_samples.shape)
        #pred_samples = pred_samples.reshape(num_batches, num_series * num_timesteps, num_samples)
        #pred_samples = pred_samples[None,:,:]
        samples = torch.zeros(num_batches, num_series * num_timesteps, num_samples, device=device)
        # print('pred_samples.shape',pred_samples.shape)
        samples[:, mask, :] = hist_true_x[:, :, None]
        # print('samples.shape',samples.shape)
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, torch.Size((num_batches, num_series, num_timesteps, num_samples)))
    