### Measurements

Sobol Set:
- [x] $z=0.0$ (snap=90) $b_1, b_2, b_{K^2}$ for all 1024 simulations
- [x] $z=0.47$ (snap=74) $b_1, b_2, b_{K^2}$ for all 1024 simulations
- [x] $z=1.04$ (snap=60) $b_1, b_2, b_{K^2}$ for all 1024 simulations

1P Variations Set:

- [x] $z=0.0$ (snap=90) 1P-set $b_1, b_2, b_{K^2}$ for all 35 parameter variations
- [x] $z=0.47$ (snap=74) 1P-set $b_1, b_2, b_{K^2}$ for all 1024 parameter variations
- [x] $z=1.04$ (snap=60) 1P-set $b_1, b_2, b_{K^2}$ for all 1024 parameter variations

### Training

Our current best training is with the relevant parameters expressed in log. However, there is still some issues with the comparison between our NPE prediction and the 1P measurements. For some of these parameters, the NPE predicts strong dependencies of the bias parameters, while in the 1P set we don't see anything.
