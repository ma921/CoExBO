# ExplainableBO

This is an ongoing project on "Explainable Bayesian Optimisation for interactive collaboration with humans".
This repo is hugely inherited from [SOBER](https://github.com/ma921/SOBER) and [GP-SHAP](https://github.com/muandet-lab/ExplainingGaussianProcess).

### Framework
- SOBER gives "choices" of next queries, like A/B test.
- GP-SHAP gives explanations of next queries.
- Human decides which choices to select.

### Learning each other
- Human learns from BO via explanation of why they recommend the choices.
- BO learns from Humans via their feedback as preferential learning.

### interact each other
- Human can rectify the BO policy via choice.
- BO can limit the choices based on the current estimate of global optimum locations.

### libraries
- gpytorch
- botorch
- qpsolvers

## Projects link
- Overleaf: https://www.overleaf.com/4616271596kscwrkqppxjt
- Notion: https://www.notion.so/bayesxl/Explainable-BO-84e3660dab3349f7a461ddb9410fbb87?pvs=4
