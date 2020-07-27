# How to Use Regression Wrappers
## Table of Contents
  - [GPR regressor](#GPR_regressor)
  - [Ridge regressor](#Ridge_regressor)

## GPR regressor
Example code of using GPR regressor with RBF kernel
```py
from prm.analyze.regressionWrapper import GPRWrapper

# error parameter determine the value added to the diagonal of the kernel matrix during fitting, large values correspond to increased noise levels
error_parameters = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

# create length scale selections of RBF kernel
kernel_parameters = GPRWrapper.create_kernel_parameters(x)

regressor = GPRWrapper.build_model(x_array, y_array, error_parameters, kernel_parameters, cv_size = 0, bootstrap_size = int(len(y_array) / 2), bootstrap_runs = 20, full_training = True, optimizer = "fmin_l_bfgs_b", fit_std = 1, print_log = 0)

y_test, _ = regressor.predict(x_test)
```

## Ridge regressor
Example code of using Ridge regressor with polynomial features

```py
from prm.analyze.ridgeWrapper import RidgeWrapper

regressor = RidgeWrapper.build_model(x_array, y_array, None, None, cv_size = 0, bootstrap_size = int(len(y_array) / 2), bootstrap_runs = 10, full_training = True, print_log = 0)

y_test, _ = regressor.predict(x_test)

```