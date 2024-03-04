# CoExBO
Source code for the AISTATS 2024 paper<br>
"Looping in the Human: Collaborative and Explainable Bayesian Optimization" [arXiv](https://arxiv.org/abs/2310.17273)

## DEMO for practitioners/researchers
We prepared an example of CoExBO with battery example. <br>
- Demo1 human feedback for battery experiments.ipynb
- Demo2 synthetic human response.ipynb

## CoExBO in a nutshell.
![plot](./docs/concept.png)<br>

Collaborative and Explainable BO (CoExBO)<br>
1. BO combines experimental results and expert preferences.<br>
2. BO generates pairwise candidates along with explanations.<br>
3. Human interprets the acquisitions and picks their preferred candidate<br>
4. Human conducts experiments and repeat step 1.<br>

# Explainability
![plot](./docs/idea.png)<br>

Utilising GP-SHAP, we can provide insights into the undergoing of the BO by attributing feature importance to the followings:
- Surrogate GP model
- Acquisition function (GP-UCB)

# Dependencies
botorch 0.8.4
gpytorch 1.10
torch 1.13.0

## Cite as
Please cite this work as
```
@inproceedings{adachi2023looping,
  title={Looping in the Human: Collaborative and Explainable Bayesian Optimization},
  author={Adachi, Masaki and Planden, Brady and Howey, David A and Maundet, Krikamol and Osborne, Michael A and Chau, Siu Lun},
  booktitle={Artificial intelligence and statistics},
  year={2024}
}
```
