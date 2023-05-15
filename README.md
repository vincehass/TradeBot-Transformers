# TradeBot-Transformers
TradeBot algorithm for sequential prediction based on Transformers and RNN Model, the main objectif of the model is to optimize portfolio allocation given uncertainity in the energy market.



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
