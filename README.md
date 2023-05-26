## Zero-inflated Negative Binomial Model using TensorFlow

TensorZINB is a Python module that uses TensorFlow for solving zero-inflated negative binomial (ZINB) and negative binomial (NB) models. It can precisely compute the ZINB log likelihood. It can also be used for differentially expressed gene (DEG) analysis for single-cell RNA sequencing (scRNA-seq) as well. The package is numerically stable, can run datasets in batches, and its computing speed is superior over other existing ZINB solvers. The analysis results from TensorZINB have been thoroughly tested against existing statistical packages to ensure that they are correct. TensorZINB supports running different features on the negative binomial part and the zero-inflated (logit) part, while common features with the same weight across all subjects can also be used in a batch.

The negative binomial distribution is
$$NB(y;\mu,\theta)=\frac{\Gamma(y+\theta)}{\Gamma(\theta)\Gamma(y+1)}\left( \frac{\theta}{\theta+\mu}\right)^\theta\left(\frac{\mu}{\theta+\mu}\right)^y$$
where $\mu$ is the mean and $\theta$ is the dispersion parameter. For zero-inflated models, the counts are modelled as a mixture of the Bernoulli distribution and count distribution, i.e.,
$$Pr(Y=0)=\pi+(1-\pi)NB(0),Pr(Y=y)=(1-\pi)NB(y),\,y>0.$$



## Installation

After downloading this repo, `cd` to the directory of downloaded repo and run:

`python setup.py install`

or 

`pip install .`

For Apple silicon (M1, M2 and etc), it is recommended to install TensorFlow by following the command in Troubleshooting section below.

## Model Estimation

`TensorZINB` solves the zero-inflated negative binomial (ZINB) and negative binomial (NB) models with given read counts. 

### Model initialization

``` r
TensorZINB(
    endog,                     # counts data: number of samples x number of subjects
    exog,                      # observed variables for the negative binomial part
    exog_c=None,               # common observed variables across all subjects for the nb part
    exog_infl=None,            # observed variables for the logit part
    exog_infl_c=None,          # common observed variables across all subjects for the logit part
    same_dispersion=False,     # whether all subjects use the same dispersion
    nb_only=False,             # whether negative binomial only without logit or zero-inflation part
)        
```

### Model fit

``` r
TensorZINB.fit(
    init_weights={},          # initial model weights. If empty, init_method is used to find init weights
    init_method="poi",        # initialization method: `poi` for Poisson and `nb` for negative binomial
    device_type="CPU",        # device_type: `CPU` or `GPU`
    device_name=None,         # None or one from `tf.config.list_logical_devices()`
    return_history=False,     # whether return loss and weights history during training
    epochs=5000,              # maximum number of epochs to run
    learning_rate=0.008,      # start learning rate
    num_epoch_skip=3,         # number of epochs to skip learning rate reduction
    is_early_stop=True,       # whether use early stop
    min_delta_early_stop=0.05,# minimum change in loss to qualify as an improvement
    patience_early_stop=50,   # number of epochs with no improvement after which training will be stopped
    factor_reduce_lr=0.8,     # factor by which the learning rate will be reduced
    patience_reduce_lr=10,    # number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.001,             # lower bound on the learning rate
    reset_keras_session=False,# reset keras session at the beginning
)        
```

### Model results

``` r
{
    "llf_total":              # sum of log likelihood across all subjects
    "llfs":                   # an array contains log likelihood for each subject
    "aic_total":              # sum of AIC across all subjects
    "aics":                   # an array contains AIC for each subject
    "df_model_total":         # total degree of freedom of all subjects
    "df_model":               # degree of freedom for each subject
    "weights":                # model weights
    "cpu_time":               # total computing time for all subjects  
    "num_sample":             # number of samples
    "epochs":                 # number of epochs run
    "loss_history":           # loss history over epochs if return_history=True
    "weights_history":        # weights history over epochs if return_history=True
}     
```

## DEG Analysis

`LRTest` provides utility for scRNA-seq DEG analysis. It runs the likelihood ratio test (LRT) by computing the log likelihood difference with and without conditions being added to the model.

To construct a `LRTest` object, we use
``` r
LRTest(
    df_data,                 # count data frame. columns: subjects (genes), rows: samples
    df_feature,              # feature data frame. columns: features, rows: samples
    conditions,              # list of features to test DEG, e.g., diagnosis
    nb_features,             # list of features for the negative binomial model
    nb_features_c=None,      # list of common features for the negative binomial model
    infl_features=None,      # list of features for the zero inflated (logit) model
    infl_features_c=None,    # list of common features for the zero inflated (logit) model
    add_intercept=True,      # whether add intercept. False if df_feature already contains intercept
    nb_only=False,           # whether only do negative binomial without zero inflation
    same_dispersion=False,   # whether all subjects use the same dispersion
)        
```

We then call `LRTest.run` to run the likelihood ratio test
``` r
LRTest.run(
    learning_rate=0.008,     # learning rate
    epochs=5000,             # number of epochs run
)        
```

The `LRTest.run` returns a result dataframe `dfr` with columns:
``` r
[
	"ll0":                   # log likelihood without conditions
	"aic0":                  # AIC without conditions
	"df0":                   # degree of freedom without conditions
	"cpu_time0":             # computing time for each subject without conditions
	"ll1":                   # log likelihood without conditions
	"aic1":                  # AIC with conditions
	"df1":                   # degree of freedom with conditions
	"cpu_time1":             # computing time for each subject with conditions
	"lld":                   # ll1 - ll0
	"aicd":                  # aic1 - aic0
	"pvalue":                # p-value: 1 - stats.chi2.cdf(2 * lld, df1 - df0)
]      
```


`tensorzinb.utils` provides utility functions:

- `normalize_features`: normalize scRNA-seq features by removing the mean and scaling to unit variance.
- `correct_pvalues_for_multiple_testing`: correct pvalues for multiple testing in Python, which is the same as `p.adjust` in `R`.

We can further correct pvalues for multiple testing by calling `correct_pvalues_for_multiple_testing(dfr['pvalue'])`.
 
## Example

An example code to show how to use `TensorZINB` and `LRTest` to perform DEG analysis can be found at `examples/deg_example.ipynb`. The example runs DEG analysis on a sample dataset with 17 clusters and 20 genes in each cluster. 


## Tests

In `tests/tensorzinb.ipynb`, we show several tests:

- validate the Poisson weights initialization.
- compare with `statsmodels` for negative binomial model only without zero-inflation to make sure the results match.
- show `statsmodels` is not numerically stable for zero-inflated negative binomial. `statsmodels` can only return results when initialized with TensorZINB results. TensorZINB results match the true parameters used to generate the samples.


## Troubleshooting

To run tensorflow on Apple silicon (M1, M2, etc), install TensorFlow using the following:

`conda install -c apple tensorflow-deps`

`python -m pip install tensorflow-macos==2.9.2`

`python -m pip install tensorflow-metal==0.5.1`


## References
Cui, T., Wang, T. A Comprehensive Assessment of Hurdle and Zero-inflated Models for Single Cell RNA-sequencing Analysis (2023).

## Support and Contribution
For technical issues particular to this repo, please report the issue on this GitHub repository.