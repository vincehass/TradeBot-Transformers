
# Quick Start

First clone this repository to your computer.

To Ensure your research is reproducible. If you don't already have it, download a conda distribution from:
https://conda.io/docs/user-guide/install/index.html.

Create a virtual environment by running:

```bash
conda create -n env_TradeBot python=3.9
source activate env_TradeBot
pip install -r requirements.txt
```



# Trading on Energy market with a convex optimization problem 

We construct a TradeBot algorithm for sequential prediction based on Transformers and RNN Model, the main objectif of the model is to optimize portfolio allocation given uncertainity in the energy market.

Our methodology is based on the following articles:

[Probabilistic forecasting with Factor Quantile Regression: Application to electricity trading](https://arxiv.org/pdf/2303.08565.pdf)

[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/pdf/2106.13008.pdf)

[Foundations of Sequence-to-Sequence Modeling for Time Series](https://arxiv.org/pdf/1805.03714.pdf)

[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf)

[Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)

[cVaR](https://bjerring.github.io/equity/2019/11/04/Portfolio-Optimization-using-CVaR.html)

[Expected shortfall](https://en.wikipedia.org/wiki/Expected_shortfall)

## Background

For this task, our goal is:

1- Develop an algorithm that synthetizes a signal of energy market that's predictive of forward returns over some time horizon.

2- Translating our signal into an algorithm that can turn a profit while also managing risk exposures, in this case is minimizing the downturn loss, or simply minimizing a constraint.

Traditionally in quantitative finance, the solution to the problem of maximizing returns while constraining risk has been to employ some form of Portfolio Optimization, but performing sophisticated optimizations is challenging on today's market.


## Identifying the Model Signal : 

Algorithmic trading strategies are driven by signals that indicate when to buy or sell assets to generate superior returns relative to 
a benchmark such as an index. The portion of an asset's return that is not explained by exposure to this benchmark is called ``alpha``, 
and hence the signals that aim to produce such uncorrelated returns are also called alpha factors.

## Recurrent Neural network (RNN)

We first adopt an RNN model that predicts a day-ahead energy prices ``da``based real time prices``rt``, in fact the trader is faced on two information:

1- the current energy prices

2- a-one-day ahead estimated prices defined by regulators, this price will let the agent to determine the maximum price (long) known as bid and the minimum prices (short) that he is willing to place.  

Since information differ from an agent to another, the bid and offer fluctuate with respect to market place (Hub node). Besides agent's descisions, there is many other factors that make prices fluctuate (quantity/price that a particular agent is forced to accept, liquidiy of the market, etc.)

In the first naive model, we consider an RNN, the major innovation of RNN is that each prediction output is a function of both previous output and new data. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences and are particularly well suited to time series forecasting. 


## Transformers (Self attention mixed with time):

In a second phase, we want to incorporate correlated market information to our model which is considered as causal effect on prices fluctuation. In terms of modeling time series data which are sequential in nature, as one can imagine, researchers have come up with models which use Recurrent Neural Networks (RNN) as discussed earlier like LSTM or GRU, and more recently Transformer based methods which fit naturally to the time series forecasting setting.

In this repo, we're going to leverage the vanilla Transformer as presented in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

The architecture is based on an Encoder-Decoder Transformer which is a natural choice for forecasting as it encapsulates several inductive biases nicely.

To begin with, the use of an Encoder-Decoder architecture is helpful at inference time where typically for some data we wish to forecast some prediction steps into the future using attention mechanism as shown in the figure below.

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/attention.png)

 We first, sample the next token which is a $24$-h of energy market time window of the $7$ hub index and pass it back into the decoder (also called ``autoregressive generation``). 
In this implementation, we use an output distribution for both the encoder and decoder and, sample from it to provide forecasts up until our desired prediction horizon. This is known as Greedy Sampling/Search, this technique will help the training step to avoid local minima but also provide uncertainty estimates for robustness.





Secondly, a Transformer helps us to train on time series data which might contain thousands of time points. It might not be feasible to input all the history of a time series at once to the model, due to the time- and memory constraints of the attention mechanism. In the gigure below, thus, one can consider some appropriate context window and sample this window and the subsequent prediction length sized window from the training data when constructing batches for stochastic gradient descent (SGD). The context sized window can be passed to the encoder and the prediction window to a causal-masked decoder. This means that the decoder can only look at previous time steps when learning the next value. This is referred to as "teacher forcing".


![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/sliding_window3.png)


In the diagram below we show how the procedure works


![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/attention_mech.png)



## The Optimization Problem: 


We consider the energy market over a short time period e.g 24hours and we want to predict the next 24 hours. We assumpe we have some amount of money to invest in any of $n$ different node energy market for every hour $i$. The optimization solution will return $v_{long}$ and $v_{short}$ combination trades per node which coresspond to (volume to long and volume to short) according to the current information (a current price $rt$ and a-day-ahead market price $da$).

The return matrix is $R$, where $R_{ij}$ is the return in dollars for the $i$-th hourly index and the $j$-th hub index. After the optimization solution we are able to retrieve the return based on the set of trades combination that coresspond to the volume to long and the volume to short for every hub index $j$. The combination of trade per dollar bet under is a subject to a constraint $C , for $v_{short}$ a $v_{long}$ with $C\geq -1000$. A betting strategy is, at time index $h$ for all node that maximize the return $R_{ij}$.

We say that there is an arbitrage opportunity in this event if there exists a betting strategy $W$ that is guaranteed to have nonnegative return for each node subject to constraint C. We can check whether there exists an arbitrage opportunity by solving the convex optimization problem.

Our optimization problem is the following:

  $$\max{\sum_{i}^\infty} hr(\textbf{v}_i^l, \textbf{p}_i^l, \textbf{v}_i^s, p_i^s,da_i, rt_i),$$ 

subject to

$\min(hr(\textbf{v}_i^l, p_i^l, v_i^s, p_i^s,da_i, rt_i))\geq -1000$ for all $i$

where $\textbf{p}^l, \textbf{v}^l, \textbf{p}^s, \textbf{v}^s=f(\textbf{x})$

$v_{ij} \geq 0, v_{ij} \geq 0, p_{ij}^l  \geq p_{ij}^s$  for all $i,j$

  
## The Startegy : Expected shortfall /CVaR

Conditional Value at Risk (CVaR) is a popular risk measure among professional investors used to quantify the extent of potential big losses. The metric is computed as an average of the  $\alpha\%$ worst case scenarios over some time horizon. 

We want to place our order/trades in a conservative way, focusing on the less profitable outcomes. For high values of $\alpha$ it ignores the most profitable but unlikely possibilities, while for small values of $\alpha$ it focuses on the worst losses. In our startegy we consider $\alpha = 5%$.

The following code implements the convex optimization based on cVaR.

```python

    def maximize_trade_constrain_downside(self,bid_price, offer_price, da_validate, rt_validate, percentile, max_loss, gamma):

        bid_return = (da_validate <= bid_price) * (rt_validate - da_validate)
        offer_return = (offer_price < da_validate) * (da_validate - rt_validate)                                                  
        
        weights1 = cp.Variable(bid_return.mean(axis=0).shape)
        weights2 = cp.Variable(offer_return.mean(axis=0).shape)
        
        objective = cp.Maximize(weights1* bid_return.mean(axis=0)+ weights2* offer_return.mean(axis=0))
        
        

        nsamples = round(bid_return.shape[0]*self.percentile)
        
        portfolio_rets = weights1*bid_return.T + weights2*offer_return.T
        wors_hour = cp.sum_smallest(portfolio_rets, nsamples)/nsamples
          
        constraints = [wors_hour>=max_loss, weights1>=0, weights2>=0, cp.norm(weights2, self.l_norm) <= self.gamma,
                                                                        cp.norm(weights1, self.l_norm) <= self.gamma]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return weights1.value.round(4).ravel(), bid_return, weights2.value.round(4).ravel(), offer_return, problem.value
```
## The best strategy based on the validation dataset:

See [ValidationExperiments](https://github.com/vincehass/TradeBot-Transformers/blob/main/ValidationExperiments.ipynb)

To ensure that our startegy is robust we apply a regularization technique as shown above in the code snippet, we choose a range of 
gamma that represent the upper bound regularizer based on the L1 and L2 norm. We pick the one that:

1- Ensure a better diversification of our portfolio (Risk diversification)

2- Ensure to achieve the maximum return under the underlying constraint.

The best model is teh one that not only exhibits the maximum return but also satisfies the constraint or the downturn loss. For both model we retain the model with $\gamma = 0.8$ and L2-norm regularizer.

In the plot below we see that our strategy respects all constraints, we note that the return are heavy tails which is common in financial data.

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/RNN_cvar.png)

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/TransformersCvar.png)

We can display the trade combination, for both models the portfolio is well diversified.

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/RNN_heatmap.png)

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/TransformersHeatmap.png)



## Limitation of the method


- We should probably add a 'slippage' model for limit orders. Slippage not only refers to the calculation of a realistic price but also a realistic volume

- The slippage method also evaluates if our order is simply too big: because we can't trade more than market's volume.

- We limit ourself to market price conditions but we could have used feature data such as climate conditions, however we assume that all external information have been captured by the price fluctuations in the market.


===========================================================================



# Robust Principle Component Analysis

Implementation of robust principal component analysis pursuit based on Algorithm 1 (Principal Component Pursuit by Alternating Directions) described on page 29 in this paper:

* Candes, Emmanuel J. et al. "Robust Principal Component Analysis?" Journal of the ACM, Vol. 58, No. 3, Article 11, 2011.

You can have acces here https://arxiv.org/abs/0912.3599



## Description
The classical _Principal Component Analysis_ (PCA) is widely used for high-dimensional analysis and dimensionality reduction. Mathematically, if all the data points are stacked as column vectors of a (n, m)matrix $M$, PCA tries to decompose $M$ as

$$M = L + S,$$

where $L$ is a rank $k$ ($k<\min(n,m)$) matrix and $S$ is some perturbation/noise matrix. To obtain $L$, PCA solves the following optimization problem

$$\min{\|L\|_* + \lambda |S|_1},$$ 

subject to $L+S = M$.

given that rank($L$) <= $k$. However, the effectiveness of PCA relies on the assumption of the noise matrix $S$: $s_{i,j}$ is small and i.i.d. Gaussian. This assumption makes the PCA not robust to outliers in the data Matrix M.

To resolve this issue, Candes, Emmanuel J. et al proposed _Robust Principal Component Analysis_ (Robust PCA or RPCA). 

### The Optimization problem
The objective is to decompose $M$ in to :

1- A low-rank matrix $L$

2- A sparse matrix $S$
 

## The Application of Robust PCA to Electricity prices

Electricity prices tend to vary smoothly in response to supply and demand signals, but are subject to intermittent price spikes that deviate substantially from normal behaviour as shown below


![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/electricity.png)

Forming the price data from one commerical trading hub into a matrix $M$ with each day as a row and each hour as a column, we can consider $M$ as the combination of a low-rank matrix $L$ consisting of the normal daily market behaviour, and a sparse matrix $S$ consisting of the intermittent price spikes.

$M$ = $L + S$

Since we can only measure the market prices $M$, we wish estimate $L$ and $S$ by solving the Robust PCA problem stated above.

Minimizing the $l_1$-norm of Spike prices $S$ is known to favour sparsity while minimizing the nuclear norm of Electricity prices $L$ is known to favour low-rank matrices (sparsity of singular values). Therefore, we have two observation to make:

1- $M$ is decomposed to a low-rank matrix but not sparse $L$ and ;

2- $S$ is a sparse but not low rank matrix. 

Here $S$ can be viewed as a sparse noise matrix which accounts the intermittent fluctuation in the market. 

The Robust PCA algorithm allows the separation of sparse but outlying values from the original data as shown below

![plot](https://github.com/vincehass/TradeBot-Transformers/blob/main/results_electricity.png)



The drawback of Robust PCA algorithm is its scalability, because it is generally slow since the implementation do SVD (singular value decomposition) in the converging iterations. 
We can alternatively look at Stable PCP which is intuitively more practical since it combines the strength of classical PCA and Robust PCA. However, we should be careful on the context of the problem and the data provided.



### To use the Robust PCA algorithm

Unroll the daily values to plot the timeseries. Note the spikes we wish to separate.


```
data = pd.read_csv("Question1.csv", index_col=0, parse_dates=True)

timeseries = data.stack()
timeseries.index = timeseries.index.droplevel(1)
timeseries.plot()

M = data.values

rpca = RobustPCA(max_iter=10000)

rpca.train_pca(M)

L = rpca.get_low_rank_matrix_L()
S = rpca.get_sparse_matrix_S()


```
Here `L` and `S` are desired low rank matrix and sparse matrix that contains the spike prices.

### Contributions
Feel free to fork and develop this project. It is under MIT license.
