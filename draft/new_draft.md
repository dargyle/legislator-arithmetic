
Todos:

- wnom: covariates in main model
- wnom: initialization
- wnom: normal distribution activation
- wnom and bayes: dropout regularization
- wnom and bayes: save and load models
- wnom and bayes: orthogonality regularization
- wnom and bayes: initialization with multiple time periods

Paper outline:

- Modern computational infrastructure makes ideal point models much more robust and useful
- We implement both a bayesian irt model and a nominate model in pytorch (and in tensorflow once upon a time)
- These implementations have the following advantages:

    - Modern: Designed for modern GPU architecture
    - Statistically robust: We have standard errors!
    - Versatile: Models can be extended in useful ways
    - Extensible: Can include additional covariates in an IRT/NOMINATE framework

- Examples:

    - European Parliament (time varying)
    - US (with party in power, test for number of dimensions?)



Ideal point estimation has long been a key tool of political science research, and recent advances in neural network training has greatly increased the computational power available to fit these models. We implement both Bayesian item response and NOMINATE parameterized ideal points using neural network frameworks, specifically pytorch. The same Bayesian model can be rapidly estimated an approximate posterior using variational bayes, or can directly apply the computation power of GPUs to find an exact distribution through Monte Carlo sampling. The NOMINATE model is estimated directly and quickly using a specially setup neural network model; standard errors in this case are obtained via bootstrapping. In both cases, the models allow for dynamic (time varying) ideal points and the inclusion of additional covariates that might explain a legislators propensity to vote yes. We demonstrate the utility of this implementation on two datasets, first estimating dynamic ideal points using the universe of Eurpoean Parliamentary votes illustrating the rise of euroskeptism. Second, we estimate ideal points for the U.S. Congress including covariates, illustrating that being a member of the party in control of the chamber increases the likelihood that a legislator votes yes on a bill independent of ideology.

## Methodology

Generically, ideal point models attempt to predict a binary outcome using latent variables. While the exact parameterization of these models takes various forms (even more when considering the universe of item response theory), this unifying classification framework fits nicely with the rise of machine learning models.
