# Positive covariance reconciliation with a prescribed portfolio

This is a project research construction. The optimization of c below is within the stated one-parameter family; no global optimum over all covariance matrices is claimed.

Let F be positive definite and let w be a nonnegative portfolio with 1'w=1. Define

\[
v=Fw,\qquad q=w'Fw>0.
\]

Choose h with h_i=1 wherever w_i>0 and h_i>=1 elsewhere. Then w'h=1. The implementation uses h_i=max(1,v_i/q) on inactive coordinates. For any c>0, define

\[
S=F-\frac{vv'}q+c\,hh'.
\]

## Positive definiteness

The matrix R=F-vv'/q is positive semidefinite by the Cauchy-Schwarz inequality in the F inner product. Its null space is exactly span(w). Since h'w=1, the positive rank-one term c hh' is strictly positive on that null space. Therefore S is positive definite.

## Prescribed long-only optimum

Since Rw=0,

\[
Sw=c\,h.
\]

The gradient is constant on positive-weight coordinates and at least that constant on inactive coordinates. Together with feasibility, these are the KKT conditions for minimizing x'Sx over nonnegative x with 1'x=1. Strict positive definiteness gives a unique optimum w. Numerical checks use the existing independent CVXPY/OSQP solver and direct relative duality-gap certificates.

## Scalar choices

The determinant identity is det(S)/det(F)=c/q. If k=h'F^(-1)h, then

\[
\operatorname{tr}(F^{-1}S)=N-1+ck.
\]

The Gaussian KL objective, omitting the common factor one half, is

\[
D(S\Vert F)=\operatorname{tr}(F^{-1}S)-\log\det(F^{-1}S)-N
=-1+ck-\log(c/q).
\]

Its minimizer is c=1/k. This is the selected model's scalar rule. The reverse Gaussian KL rule gives c=q. Minimizing squared Frobenius distance gives

\[
c=\frac{(h'v)^2}{q\,(h'h)^2}.
\]

The research runner tests all three scalar rules. These choices change the covariance forecast while preserving the same desired long-only allocation.

The Frobenius expression must be strictly positive to stay in this positive-definite family. An exact zero gives a boundary infimum rather than an admissible positive-definite minimizer; the API rejects that case. The selected Gaussian-KL rule is always strictly positive.

When w is already the long-only optimum of F, the chosen h equals v/q up to numerical tolerance. Each scalar rule gives c=q, and S=F. Thus the construction preserves an already consistent forecast.

## Causal stress rule

Let C_s denote observed realized covariance matrices. For forecast date t, compute

\[
z_t=\frac{\operatorname{tr}(C_{t-1})}{\frac1{22}\sum_{j=1}^{22}\operatorname{tr}(C_{t-j})},
\qquad b_t=\operatorname{clip}\!\left(\frac{\log z_t}{\log3},0,1\right).
\]

The fast covariance target is an EWMA of strictly earlier realized matrices with decay 2^(-1/5) per retained observation. If w_F and w_fast are the corresponding long-only allocations, the desired portfolio is

\[
w_t=(1-b_t)w_F+b_t w_{fast}.
\]

It is feasible by convexity. All observed inputs precede t; the current day's realized matrix and return are not used. The result remains a covariance forecast evaluated against the same realized target, and the portfolio is recomputed using the original optimization constraints.

Positive definiteness and KKT consistency are mathematical properties. They do not guarantee lower future forecast error or portfolio risk. The featured configuration was selected retrospectively from the recorded study and requires independent evaluation.
