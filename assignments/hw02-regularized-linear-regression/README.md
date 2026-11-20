# Homework 2 - Regularized Linear Regression

Status: not started. Depends on [homework 1](../hw01-linear-regression/README.md).

Source of requirements: the instructor-provided *Linear Regression - Programming Assignment - Part 2* PDF, retained locally and excluded from the public repository. The original brief takes precedence over this checklist.

## Implementation checklist

- [ ] Reuse homework 1's dataset, source code, and, as far as possible, preprocessing and basis functions.
- [ ] Enrich the basis sufficiently to approach zero training error and demonstrate conditions for overfitting.
- [ ] Derive and implement all five approaches below.

| Label | Learning method | Regularization |
| --- | --- | --- |
| CF | Closed-form, full batch | Quadratic / Ridge |
| GD-R | Full-batch gradient descent | Quadratic / Ridge |
| GD-L | Full-batch gradient descent | L1 / Lasso |
| SGD-R | One randomly selected training sample per update | Quadratic / Ridge |
| SGD-L | One randomly selected training sample per update | L1 / Lasso |

- [ ] Use approximately 80% training / 20% testing; fix the split across all methods and lambda values.
- [ ] Include the regularization term consistently in the objective and gradients.
- [ ] Handle the L1 absolute value's nondifferentiability at zero.
- [ ] State the intercept convention; the brief permits excluding it from regularization.
- [ ] Sweep several logarithmically spaced lambda values (the brief gives 0.01, 0.1, 1, 10 as examples).
- [ ] Report test RMSE for every method and lambda in a table or plot.
- [ ] Count nonzero weights for GD-L and SGD-L at each lambda; state the numerical tolerance used.
- [ ] Discuss sparsity and interpret retained/removed basis functions in the dataset's context.
- [ ] Compare GD and SGD fairly, accounting for sample processing, parameter updates, and stopping criteria rather than iteration counts alone.

## Submission checklist

- [ ] Describe the dataset, show its first rows, and explain feature/target meanings and collection provenance.
- [ ] Explain the mathematical formulation and derivations.
- [ ] Explicitly include the derivation of GD-L parameter updates and the corresponding SGD-L updates.
- [ ] Include all source code and installation/execution instructions.
- [ ] Include experimental tables/plots, discussion, and conclusions.

The brief permits a single organized Jupyter Notebook containing the deliverables, or a publicly accessible cloud link to the complete deliverables.

## Implementation constraints

As in homework 1, implement basis functions, losses, gradients, and learning procedures ourselves. The brief explicitly permits library solvers for linear systems, inverses, and least squares, alongside basic vector/matrix operations, and strongly encourages avoiding high-level ML libraries and deep learning frameworks.
