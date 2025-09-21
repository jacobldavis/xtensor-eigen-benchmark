# Benchmarking xtensor and Eigen

Terminal output with 12th Gen Intel i7-12650H

Matrix-Vector Multiplication (100x100):
  C:       0.055218s
  Eigen:   0.005299s
  xtensor: 0.070745s

Matrix-Vector Multiplication (1000x1000):
  C:       0.088134s
  Eigen:   0.010645s
  xtensor: 0.072300s

Matrix-Vector Multiplication (5000x5000):
  C:       0.228572s
  Eigen:   0.059371s
  xtensor: 0.195243s

Element-wise Operations (exp(x²) + sin(x), n=100000):
  C:       0.885920s
  Eigen:   0.000007s
  xtensor: 0.000011s

Element-wise Operations (exp(x²) + sin(x), n=1000000):
  C:       0.886727s
  Eigen:   0.000002s
  xtensor: 0.000002s

Dot Product (n=100000):
  C:       0.044729s
  Eigen:   0.019612s
  xtensor: 0.088063s

Dot Product (n=1000000):
  C:       0.046616s
  Eigen:   0.023371s
  xtensor: 0.088582s

Cubic Interpolation (n_data=1000, n_eval=10000):
  C:       0.004085s
  Eigen:   0.006427s
  xtensor: 0.029184s

Cubic Interpolation (n_data=10000, n_eval=50000):
  C:       0.002514s
  Eigen:   0.004117s
  xtensor: 0.014358s

3D Rotation Transform (n=10000):
  C:       0.011291s
  Eigen:   0.008771s
  xtensor: 0.066439s

3D Rotation Transform (n=100000):
  C:       0.012991s
  Eigen:   0.010512s
  xtensor: 0.068951s