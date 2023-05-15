# TradeBot-Transformers
TradeBot algorithm for sequential prediction based on Transformers and LSTM Model, the main objectif of the model is to optimize portfolio allocation given uncertainity in the financial market.



# Robust Principle Component Analysis

Implementation of robust principal component analysis pursuit based on Algorithm 1 (Principal Component Pursuit by Alternating Directions) described on page 29 in this paper:

* Candes, Emmanuel J. et al. "Robust Principal Component Analysis?" Journal of the ACM, Vol. 58, No. 3, Article 11, 2011.

You can have acces here https://arxiv.org/abs/0912.3599



## Description
The classical _Principal Component Analysis_ (PCA) is widely used for high-dimensional analysis and dimensionality reduction. Mathematically, if all the data points are stacked as column vectors of a (n, m)matrix $M$, PCA tries to decompose $M$ as

$$M = L + S,$$

where $L$ is a rank $k$ ($k<\min(n,m)$) matrix and $S$ is some perturbation/noise matrix. To obtain $L$, PCA solves the following optimization problem

$$\min_{L} ||M-L||_2,$$

given that rank($L$) <= $k$. However, the effectiveness of PCA relies on the assumption of the noise matrix $S$: $s_{i,j}$ is small and i.i.d. Gaussian. This assumption makes the PCA not robust to outliers in the data Matrix M.

To resolve this issue, Candes, Emmanuel J. et al proposed _Robust Principal Component Analysis_ (Robust PCA or RPCA). 

### The Optimization problem
The objective is to decompose $M$ in to :
1- A low-rank matrix $L$
2- A sparse matrix $S$
 
Then we have an optimization problem that looks as the following

$$\min_{L,S} ||L||_{*} + \lambda||S||_{1}$$ 

subject to $L+S = M$.


## The Application of Robust PCA to Electricity prices

Electricity prices tend to vary smoothly in response to supply and demand signals, but are subject to intermittent price spikes that deviate substantially from normal behaviour.

Forming the price data from one commerical trading hub into a matrix $M$ with each day as a row and each hour as a column, we can consider $M$ as the combination of a low-rank matrix $L$ consisting of the normal daily market behaviour, and a sparse matrix $S$ consisting of the intermittent price spikes.

$M$ = $L + S$

Since we can only measure the market prices $M$, we wish estimate $L$ and $S$ by solving the Robust PCA problem:

$\min{\|L\|_* + \lambda |S|_1}$

subject to $L + S = M$ 

Minimizing the $l_1$-norm of Spike prices $S$ is known to favour sparsity while minimizing the nuclear norm of Electricity prices $L$ is known to favour low-rank matrices (sparsity of singular values). Therefore, we have two observation to make:
1- $M$ is decomposed to a low-rank matrix but not sparse $L$ and ;
2- $S$ is a sparse but not low rank matrix. 

Here $S$ can be viewed as a sparse noise matrix which accounts the intermittent fluctuation in the market. 

The Robust PCA algorithm allows the separation of sparse but outlying values from the original data.  



The drawback of Robust PCA algorithm is its scalability, because it is generally slow since the implementation do SVD (singular value decomposition) in the converging iterations. 
We can alternatively look at Stable PCP which is intuitively more practical since it combines the strength of classical PCA and Robust PCA. However, we should be careful on the context of the problem and the data provided.



### To use the Robust PCA algorithm


```
from RobustPCA.rpca import RobustPCA

rpca = RobustPCA()
<!-- spcp = StablePCP() -->

rpca.fit(M)
L = rpca.get_low_rank()
S = rpca.get_sparse()

<!-- spcp.fit(M)
L = spcp.get_low_rank()
S = spcp.get_sparse() -->
```
Here `L` and `S` are desired low rank matrix and sparse matrix.

### Contributions
Feel free to fork and develop this project. It is under MIT license.
