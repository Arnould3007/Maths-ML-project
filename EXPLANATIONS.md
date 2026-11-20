# Project walkthrough: one-hidden-layer neural network

> This file is a guide for the 1:1 oral defense.
> For each file: **what it does**, **the math behind it**, **the design choices and why**, and **likely questions from the professor**.
> It is filled in progressively as the code is written.

---

## 0. Overview

### Goal
Implement from scratch, using only NumPy, a neural network with **a single hidden layer**, in two versions:
- `SimpleSLPClassifier`: predicts a class (binary or multi-class)
- `SimpleSLPRegressor`: predicts a continuous value

Both follow the scikit-learn interface (`fit`, `predict`, `score`), so they can be compared directly with `MLPClassifier` and `MLPRegressor`.

### Network architecture

```
X  ──►  hidden layer  ──►  output layer  ──►  prediction
        z1 = X @ W1 + b1    z2 = a1 @ W2 + b2
        a1 = g(z1)          y_pred = output(z2)
```

- `g` = hidden activation function (relu, tanh or logistic)
- `output` = sigmoid (binary), softmax (multi-class) or identity (regression)

### Training loop (gradient descent)
1. **Forward pass**: compute the prediction
2. **Loss**: measure the error
3. **Backward pass**: compute the gradients with the chain rule
4. **Update**: `W = W - learning_rate * dW`
5. Repeat until `max_iter` (or early stopping)

### Link with the course material

| Course source | Formula used in the project |
|---|---|
| Session 4, slide 6 (Perceptron) | `z = Σ w_i x_i + b`, then an activation: this is one neuron. Our hidden layer is many of these in parallel: `z1 = X @ W1 + b1`, with a smooth activation instead of the step function |
| Session 4, slide 5 (Logistic regression) | `p(y=1│x) = σ(xᵀβ) = 1 / (1 + e^(-xᵀβ))`, and `∇ℓ(β) = Xᵀ(y − σ(Xβ))`. Our binary classifier output is exactly a logistic regression applied to `a1` instead of `X` |
| Session 4, slide 2 (MLE recipe) | Minimizing the cross-entropy = maximizing the log-likelihood |
| Session 3, slide 17 | Cross-entropy `H(p, q) = −Σ p(x) log q(x)` |
| Session 3, slide 29 | Binary: `E(w) = −Σ_n {t_n ln y_n + (1 − t_n) ln(1 − y_n)}`. Multi-class: `E = −Σ_n Σ_k t_nk ln y_nk`. Gradient w.r.t. the output pre-activation: `(y_n − t_n)`. Regularized: `E_R(w) = E(w) + (λ/2)‖w‖²` |
| Session 2, slide 6 | Jacobian / gradient definitions used by the chain rule in backpropagation |
| Session 6, slides 15-16 | "Build an MLP from scratch with NumPy": architecture, forward, backprop, gradient descent, test-set evaluation |
| `ExerciseSolution.ipynb` | Sigmoid coded as `1 / (1 + np.exp(-z))`; `log(p + 1e-10)` to avoid `log(0)` |

Notation: in the course, `t` = true target and `y` = prediction. In the code, `y_true` (or `y`) = target and `y_pred` = prediction.

### Code structure

| File | Role |
|---|---|
| `activations.py` | Activation functions and their derivatives |
| `slp_base.py` | Shared parent class (parameters, weight initialization) |
| `slp_classifier.py` | Network for classification |
| `slp_regressor.py` | Network for regression |
| `test_*.py` | Tests provided by the professor |

---

## 1. `activations.py`

### What this file does
It contains the **activation functions** and their **derivatives**. Every function takes a NumPy array `z` and returns an array of the same shape, computed element by element.

- **Activations** (`relu`, `tanh`, `logistic`) are used in the **forward pass**, on the hidden layer: `a1 = g(z1)`.
- **Derivatives** are used in the **backward pass**: the chain rule needs `g'(z1)` to send the error back from the output to the hidden layer.
- **Softmax** is only used on the **output layer** for multi-class classification.

### Why do we need an activation function at all?
Without one, the network would be `y = (X @ W1 + b1) @ W2 + b2`, which simplifies to `X @ W + b`: a plain **linear model**, no matter how many layers. The activation adds **non-linearity**, so the network can learn curved decision boundaries (e.g. the XOR problem, which no straight line can separate).

### Why are all functions vectorized?
Each function uses NumPy operations (`np.maximum`, `np.exp`, `np.tanh`...) that apply to the whole array at once, with no Python `for` loop. This is much faster and is one of the "advanced requirements" of the project.

---

### ReLU (Rectified Linear Unit)

| | Formula | Code |
|---|---|---|
| Function | `ReLU(z) = max(0, z)` | `np.maximum(0, z)` |
| Derivative | `1 if z > 0, else 0` | `(z > 0).astype(float)` |

- Negative inputs become 0, positive inputs pass through unchanged.
- **Pros:** very cheap to compute, and its gradient is 1 for positive values, so it does not "shrink" the gradient (no vanishing gradient on the positive side). It is the default choice in modern deep learning.
- **Cons:** a neuron whose input is always negative has gradient 0 and stops learning ("dying ReLU").
- **Choice at z = 0:** ReLU is not differentiable at 0. By convention we use 0 (this is what the tests expect), hence `z > 0` and not `z >= 0`.
- **`np.maximum` vs `np.max`:** `np.maximum(0, z)` compares each element with 0. `np.max(z)` would return a single number (the largest value of the array).

---

### tanh (hyperbolic tangent)

| | Formula | Code |
|---|---|---|
| Function | `tanh(z) = (e^z - e^-z) / (e^z + e^-z)` | `np.tanh(z)` |
| Derivative | `1 - tanh(z)^2` | `1 - np.tanh(z) ** 2` |

- Output in **(-1, 1)**, centered on 0. A zero-centered output usually helps gradient descent converge faster than the sigmoid.
- The derivative is maximal (= 1) at z = 0 and goes to 0 for large |z|: the neuron "saturates" and learns slowly (**vanishing gradient**).

---

### Logistic / sigmoid

| | Formula | Code |
|---|---|---|
| Function | `σ(z) = 1 / (1 + e^-z)` | `1 / (1 + np.exp(-z))` |
| Derivative | `σ(z) · (1 - σ(z))` | `s * (1 - s)` with `s = logistic(z)` |

- Output in **(0, 1)**, which can be read as a **probability**. That is why it is also the output activation for **binary classification**.
- The derivative is at most **0.25** (at z = 0). In deep networks, multiplying many values ≤ 0.25 makes the gradient vanish, which is why ReLU is often preferred in hidden layers.
- **Numerical stability:** for z = -1000, `np.exp(1000)` overflows to infinity. We **clip** z to [-500, 500] first. This does not change the result in practice: σ(500) is already exactly 1.0 in floating point.
- **Derivative proof (short):** σ(z) = (1 + e^-z)^-1, so σ'(z) = e^-z / (1 + e^-z)^2 = σ(z) · (1 - σ(z)).
- **Code reuse:** `logistic_derivative` calls `logistic` instead of rewriting the formula (one single source of truth, and the clipping is applied too).
- **Same as the course:** the formula is written exactly like in `ExerciseSolution.ipynb` (`1 / (1 + np.exp(-z))`) and the derivative like in `Script3.ipynb` (`sigma * (1 - sigma)`). The only addition is the `np.clip`, requested by the hint in the scaffold.

---

### Softmax

| | Formula | Code |
|---|---|---|
| Function | `softmax(z)_i = e^(z_i) / Σ_j e^(z_j)` | see below |

```python
z_shifted = z - np.max(z, axis=1, keepdims=True)
exp_z = np.exp(z_shifted)
return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

- Turns a vector of scores (one per class) into **probabilities**: all positive, summing to 1. The largest score gets the largest probability. It is the generalization of the sigmoid to K classes.
- Input shape: `(n_samples, n_classes)`. Each **row** is one sample, so all operations are done **per row** (`axis=1`).
- **`keepdims=True`:** `np.max(z, axis=1)` would return shape `(n_samples,)`; with `keepdims=True` it returns `(n_samples, 1)`, which NumPy can **broadcast** to subtract from every column of the row.
- **Max-subtraction trick:** for z = [1000, 1001, 1002], `np.exp(1000)` overflows. Subtracting a constant c from every score does not change the result:
  `e^(z_i - c) / Σ e^(z_j - c) = e^-c · e^(z_i) / (e^-c · Σ e^(z_j)) = e^(z_i) / Σ e^(z_j)`.
  With c = max(z), the largest exponent is e^0 = 1, so no overflow is possible.
- **No derivative function:** with a cross-entropy loss, the gradient of "softmax + cross-entropy" simplifies to `y_pred - y_true`, so the softmax derivative alone is never needed (same for sigmoid + binary cross-entropy).

---

### Tests
`test_activations.py` checks every function on hand-verifiable values (e.g. σ(0) = 0.5, σ'(0) = 0.25, ReLU'(0) = 0, softmax of equal scores = uniform, softmax of [1000, 1001, 1002] has no NaN/inf).
Result: **21/21 tests pass**.

---

## 2. `slp_base.py`

### What this file does
`BaseSLPEstimator` is the **parent class** of the classifier and the regressor. It holds everything the two models share:
- the hyperparameters (`hidden_layer_size`, `activation`, `learning_rate`, `max_iter`, `random_state`, plus `batch_size` and the early-stopping options),
- the choice of the activation function (`_get_activation_function`),
- parameter and input validation (`_validate_params`, `_validate_input`),
- weight initialization (`_initialize_weights`),
- the **training loop** (`_train`): mini-batch gradient descent + early stopping.

What differs between the two models (output activation, loss, gradient of the output layer, `predict`, `score`) is declared as `@abstractmethod` and written in each subclass. This is the **template method** pattern: the parent defines the algorithm, the children fill in the specific steps.

### Shapes used everywhere

| Symbol | Meaning | Array | Shape |
|---|---|---|---|
| n | number of samples | `X` | `(n, d)` |
| d | number of features (`n_features_in_`) | `W1` | `(d, h)` |
| h | hidden neurons (`hidden_layer_size`) | `b1` | `(h,)` |
| k | output neurons (`n_outputs_`) | `W2` | `(h, k)` |
| | | `b2` | `(k,)` |

`n_features_in_` and `n_outputs_` are the scikit-learn names. They are set in `fit()` (they depend on the data) **before** `_initialize_weights()` is called, because that method has no arguments in the scaffold.

---

### `_initialize_weights`

```python
self._rng = np.random.default_rng(self.random_state)
d, h, k = self.n_features_in_, self.hidden_layer_size, self.n_outputs_
self.W1_ = self._rng.normal(loc=0.0, scale=np.sqrt(2.0 / (d + h)), size=(d, h))
self.b1_ = np.zeros(h)
self.W2_ = self._rng.normal(loc=0.0, scale=np.sqrt(2.0 / (h + k)), size=(h, k))
self.b2_ = np.zeros(k)
```

- **Weights ~ N(0, 2 / (fan_in + fan_out))**: Glorot/Xavier initialization, the one scikit-learn uses.
  - **Random**, to break the symmetry: if all weights were equal (e.g. 0), every hidden neuron would compute the same value, receive the same gradient and stay identical forever. The hidden layer would behave like a single neuron.
  - **Scaled to the layer size**, so that the variance of the signal stays roughly the same from one layer to the next (forward) and so does the variance of the gradient (backward). With 10 inputs and 30 hidden neurons, std = √(2/40) ≈ 0.22.
- **Why not the scaffold's example (std 0.01)?** It was the first version, and the classifier did not learn: with `W2 ≈ 0.01`, the gradient sent to the hidden layer (`da1 = dz2 @ W2.T`) is ~100 times too small, and with a sigmoid `a1 ≈ 0.5` for every sample, so the hidden layer carries no information (a **plateau**). Measured on the classifier test dataset (logistic, `learning_rate=0.001`, 200 epochs): accuracy **0.52 with std 0.01 vs 0.70 with Glorot**. The scaffold says "e.g.", so another initialization is allowed. The regressor tests all still pass with Glorot.
- **Biases = 0**: the random weights are already enough to break the symmetry.
- **`np.random.default_rng(random_state)`** instead of `np.random.seed(...)`: it creates a *local* generator. Same seed gives the same weights (reproducibility, checked by `test_reproducibility`), and it does not change NumPy's *global* random state, which the user's own code may rely on. The generator is kept in `self._rng` and reused to shuffle the mini-batches, so the whole training run is reproducible from one seed.

---

### `_validate_params` and `_validate_input`
- `_validate_params` (called at the start of `fit`) raises a clear `ValueError` for impossible settings: `hidden_layer_size < 1`, `learning_rate <= 0`, unknown activation, `batch_size < 1`, `validation_fraction` outside (0, 1), etc. Failing early with a clear message is better than a cryptic NumPy error 200 lines later.
- `_validate_input` converts `X` to a 2D float array (so lists work too), checks that `X` and `y` have the same number of rows and, at prediction time, that `X` has the same number of features as during `fit`.

---

### `_train`: mini-batch gradient descent + early stopping

**One epoch** (= one iteration of `max_iter`, like `MLPRegressor`):
1. shuffle the sample indices (`self._rng.permutation(n)`),
2. cut them into mini-batches of `batch_size` samples,
3. for each batch: forward, backward, then the update `θ ← θ − η · ∂L/∂θ` for θ ∈ {W1, b1, W2, b2},
4. compute the loss on the whole training set and append it to `loss_curve_`.

The only `for` loops are over **epochs** and **batches**, never over samples or weights: inside a batch everything is matrix algebra (vectorized).

**Why mini-batches (`batch_size=32` by default) and not full batch?** Measured on the regressor test datasets (`test_linear_function` and the `regression_data` fixture: 30 % test split, `random_state=42`, raw y, R² on the test set, `learning_rate=0.01`, 300 epochs, Glorot initialization):

| | linear / logistic | linear / relu | make_regression / logistic | make_regression / relu |
|---|---|---|---|---|
| full batch | **0.881** | 0.989 | 0.926 | 0.985 |
| batch 32 | **0.981** | 0.998 | 0.932 | 0.990 |
| batch 16 | 0.985 | 0.999 | 0.926 | NaN (diverges) |

Full batch makes only 300 updates in 300 epochs. With batches of 32 on 140 samples there are 5 updates per epoch (1 500 in total) for the same computing cost, so with the same learning rate the model gets further: the slowest case (sigmoid hidden layer on the linear target) goes from 0.881 to 0.981. Batch 16 makes steps too large for targets of order 100 and diverges. `batch_size=None` still gives full-batch gradient descent.
*(With the first initialization, std 0.01, full batch was completely stuck on a plateau: R² = 0.003 on the linear/logistic case. Glorot removed the plateau, but mini-batches still converge faster.)*

**Sum or mean: why the step size depends on the model.** The regressor's loss is a **mean** (`1/(2n)`), so its gradient does not depend on the batch size. The classifier's loss is a **sum** (slide 29), so its gradient grows with the batch size: one step of `learning_rate` on a batch of size B is the same as a step of `learning_rate × B` on the mean loss. That is why the classifier works with a smaller `learning_rate` (0.001), and why full batch (B = n) with a large learning rate can diverge for the classifier (breast cancer, `learning_rate=0.1`, `batch_size=None`: NaN after 89 epochs, while `batch_size=32` reaches 0.965 accuracy).

**Early stopping (`early_stopping=True`)**, same logic as scikit-learn:
- a random `validation_fraction` (10 %) of the training data is held out and **never** used for gradient steps,
- after each epoch, the loss on this validation set is computed (`validation_loss_curve_`),
- if it has not decreased by at least `tol` for `n_iter_no_change` epochs in a row, training stops,
- the weights of the **best** validation epoch are restored (`.copy()` is needed, otherwise we would save a reference to arrays that keep changing).

It stops training when the model starts to **overfit** (training loss still goes down, validation loss goes up), and saves time. Example on `load_diabetes` (80/20 split, `random_state=42`, standardized X and y, 30 hidden neurons, relu, `learning_rate=0.1`, `max_iter=500`): R² = 0.127 without early stopping vs **0.477** with it (stopped at epoch 15/500).
With `learning_rate=0.01` there is little overfitting to prevent: early stopping saves 70-95 % of the epochs for almost the same R² (relu 0.474 → 0.468, logistic 0.469 → 0.467, tanh 0.471 → 0.441). It costs 10 % of the training data (the validation set) and adds randomness (which samples are held out), so it is off by default, as in scikit-learn.
**`tol` and the summed loss:** `tol` is an absolute decrease of the validation loss. For the regressor (mean loss) it is a decrease *per sample*; for the classifier (summed loss) it is a decrease of the total over the validation set, so per sample it is about `n_val` times stricter (≈ 46 times on breast cancer). For the same reason, `validation_loss_curve_` and `loss_curve_` of the classifier are sums over different numbers of samples: divide each by its own n before plotting them together.

**Divergence guard**: if the loss becomes `inf`/`NaN` (learning rate too large, or unscaled target), training stops with a `RuntimeWarning` that says what to change, and the weights of the last finite epoch are restored, so the model never keeps NaN weights. This follows the "monitor numerical stability" tip of the project.

---

## 3. `slp_regressor.py`

### What this file does
`SimpleSLPRegressor` predicts a **continuous value**. It inherits everything shared from `BaseSLPEstimator` and implements the regression-specific parts: identity output, MSE loss, its gradient, `fit`, `predict` and the R² `score`.

### Forward propagation

```python
z1 = X @ self.W1_ + self.b1_   # (n, d) @ (d, h) + (h,) -> (n, h)
a1 = activation(z1)            # (n, h)
z2 = a1 @ self.W2_ + self.b2_  # (n, h) @ (h, k) + (k,) -> (n, k)
y_pred = z2                    # identity output
```

- `b1` has shape `(h,)` and is added to every row of `X @ W1` by **broadcasting**: every sample gets the same bias.
- **Identity output**: a regression target can be any real number. A sigmoid would force the output into (0, 1) and could never predict, say, 150.
- The function returns `z1, a1, z2, y_pred` (not only `y_pred`) because backpropagation needs these intermediate values: they are computed once and reused.

### Loss: mean squared error

$$L = \frac{1}{2n}\sum_{i=1}^{n}\sum_{j=1}^{k}(\hat y_{ij} - y_{ij})^2$$

```python
return float(np.sum((y_pred - y_true) ** 2) / (2 * n_samples))
```

- The **½** cancels the 2 coming from the derivative of the square, so the gradient is exactly `(y_pred − y)/n`. Without it, the gradient would be `2(y_pred − y)/n`: same minimum, but it acts like a learning rate twice as large.
- **Loss and gradient must use the same factor.** If they did not, `loss_curve_` would describe a different function from the one gradient descent is actually minimizing.
- MSE = maximum likelihood under Gaussian noise (Session 4, slide 2, MLE recipe): if `y = f(x) + ε` with `ε ~ N(0, σ²)`, maximizing the log-likelihood is the same as minimizing Σ(y − f(x))².

### Backpropagation (chain rule)

```python
dz2 = (y_pred - y) / n_samples          # (n, k)
dW2 = a1.T @ dz2                        # (h, n) @ (n, k) -> (h, k)
db2 = dz2.sum(axis=0)                   # (k,)
da1 = dz2 @ self.W2_.T                  # (n, k) @ (k, h) -> (n, h)
dz1 = da1 * activation_derivative(z1)   # (n, h)
dW1 = X.T @ dz1                         # (d, n) @ (n, h) -> (d, h)
db1 = dz1.sum(axis=0)                   # (h,)
```

Line by line:
1. **`dz2 = ∂L/∂z2`**. With an identity output, `y_pred = z2`, so `∂L/∂z2 = ∂L/∂y_pred = (y_pred − y)/n`. This is the **output-layer error**, the same form `(y_n − t_n)` as Session 3, slide 29.
2. **`dW2`**: `z2 = a1 @ W2 + b2`, so the weight `W2[p, j]` is multiplied by `a1[i, p]` for every sample i. Its gradient is `Σᵢ a1[i, p] · dz2[i, j]`, which is exactly the matrix product `a1.T @ dz2`. **The matrix product does the sum over samples.**
3. **`db2`**: the bias is added to every sample, so its gradient is the sum of `dz2` over the samples (`axis=0`).
4. **`da1`**: send the error back through the weights: each hidden neuron receives the output errors weighted by its outgoing weights (`W2.T`).
5. **`dz1`**: chain rule through the activation `a1 = g(z1)`, applied element by element, so it is `*` and **not** `@`. This is where `relu_derivative`, `tanh_derivative` and `logistic_derivative` from `activations.py` are used.
6. **`dW1`, `db1`**: same as steps 2-3, one layer lower (with `X` in place of `a1`).

**Shape check (useful at the oral):** every gradient has **the same shape as the parameter it updates**. That is how you know where `.T` goes.

**Verification:** the analytical gradients were compared with finite differences `(L(θ+ε) − L(θ−ε)) / 2ε` for every parameter and all 3 activations: max error ~1e-10, so the backpropagation is correct.

### `fit`

```python
self._validate_params()
X, y = self._validate_input(X, y)
y = y.astype(float)
self._y_is_1d = y.ndim == 1
Y = y.reshape(-1, 1) if self._y_is_1d else y
self.n_features_in_ = X.shape[1]
self.n_outputs_ = Y.shape[1]
self._train(X, Y)
return self
```

- Inside the network, `y` is always 2D `(n, k)`, so that `y_pred - y` is `(n, 1) - (n, 1)`. If `y` stayed `(n,)`, then `(n, 1) - (n,)` would **broadcast** to an `(n, n)` matrix: no error, but a completely wrong gradient. This is a classic NumPy trap.
- `_y_is_1d` remembers the original format so that `predict` returns the same shape as the `y` given to `fit` (the tests expect `(n,)`).
- `return self` is the scikit-learn convention (it allows `model.fit(X, y).predict(X)`).
- The training loop itself is the shared `_train` from the base class.

### `predict`
Forward pass only (no gradient), then `ravel()` back to `(n,)` if `y` was 1D. It raises a clear error if the model has not been fitted, or if `X` has the wrong number of features.

### `score`: R² (coefficient of determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

- `SS_tot` is the error of the dumbest model, which always predicts the mean ȳ.
- R² = 1 means perfect predictions. R² = 0 means no better than the mean. **R² < 0** means worse than the mean (possible on a test set).
- For several outputs, R² is computed per output and then averaged, exactly like `sklearn.metrics.r2_score` (checked: identical values).

### Target scaling (important for the report)
The model does **not** standardize `y` internally:
- `MLPRegressor` does not standardize `y` either: we keep the same interface and behaviour, and leave the preprocessing choice to the user (like `StandardScaler` for X);
- the tests pass with raw `y` (with Glorot, standardizing y would also work: R² = 0.994 instead of 0.981 on the linear test, so this is a design choice, not a necessity);
- but on `load_diabetes` (y between 25 and 346), raw `y` makes `dz2` huge and relu **diverges** (the loss becomes inf/NaN during the first epoch). Even sklearn's `MLPRegressor` only reaches R² = −0.35 on the raw target.

So, as the project tips say, **standardize the target yourself** for real datasets and transform the predictions back:

```python
mu, sd = y_train.mean(), y_train.std()
reg.fit(X_train, (y_train - mu) / sd)
y_pred = reg.predict(X_test) * sd + mu
```

On `load_diabetes` (80/20 split, `random_state=42`, standardized X and y, 30 hidden neurons, `learning_rate=0.01`, 500 epochs): **R² = 0.474 (relu), 0.471 (tanh), 0.469 (logistic) for our model vs 0.495 for `MLPRegressor`** (default settings, Adam). Same order of magnitude; Adam's adaptive steps give sklearn a small edge. The full comparison is in `report.ipynb`.

### Tests
`pytest -q test_models.py -k "Regressor or regressor"`: **8/8 pass** (initialization, output shape and finiteness, R² > 0.5, loss decreases, constant target, linear function R² > 0.8, minimal data, comparison with sklearn R² > 0.6). The regressor smoke test passes too.

---

## 4. `slp_classifier.py`

### What this file does
`SimpleSLPClassifier` predicts a **class**. It reuses the shared `_train` loop of the base class and implements the classification-specific parts: label encoding, sigmoid/softmax output, cross-entropy loss, its gradient, `predict_proba`, `predict` and the accuracy `score`. The hidden layer is exactly the same as in the regressor; only the **output layer** and the **loss** change.

| | Regressor | Classifier (binary) | Classifier (multi-class) |
|---|---|---|---|
| Output neurons `k` | 1 | 1 | number of classes K |
| Output activation | identity | sigmoid | softmax |
| Loss | MSE | binary cross-entropy | categorical cross-entropy |
| Output gradient `dz2` | `(y_pred − y)/n` | `y_pred − y` | `y_pred − y` |

### `fit`: encoding the labels

```python
self.classes_ = np.unique(y)
if n_classes <= 2:
    Y = (y == self.classes_[-1]).astype(float).reshape(-1, 1)   # (n, 1)
else:
    Y = (y.reshape(-1, 1) == self.classes_).astype(float)       # (n, K)
self.n_features_in_ = X.shape[1]
self.n_outputs_ = Y.shape[1]
self._train(X, Y)
```

- `np.unique` gives the **sorted** list of labels, e.g. `[0, 1, 2]`. It works with any labels (`["cat", "dog"]`, `[3, 7]`), not only 0/1: the network works with column indices, and `classes_` translates them back.
- **Binary**: a single output neuron is enough, because P(class 0) = 1 − P(class 1). The target is 1 for the last class, 0 otherwise.
- **Multi-class: one-hot encoding.** Each label becomes a row with a single 1: with classes `[0, 1, 2]`, label 2 → `[0, 0, 1]`. The comparison `y.reshape(-1, 1) == self.classes_` is `(n, 1)` against `(K,)`, which **broadcasts** to an `(n, K)` table of True/False: one-hot encoding in one line, without a loop.
- **A single class** (`test_all_same_class`): handled by the binary branch; `predict_proba` then returns a column of 1 and `predict` always returns that class.

### Forward propagation

```python
z1 = X @ self.W1_ + self.b1_
a1 = activation(z1)
z2 = a1 @ self.W2_ + self.b2_
y_pred = logistic(z2) if self.n_outputs_ == 1 else softmax(z2)
```

- **Sigmoid** (binary): turns the score `z2` into a probability in (0, 1). This output layer is exactly a **logistic regression** applied to `a1` instead of `X` (Session 4, slide 5).
- **Softmax** (multi-class): turns K scores into K probabilities that sum to 1 (see section 1).

### Loss: cross-entropy (Session 3, slide 29)

Binary: $E(w) = -\sum_n \left[t_n \ln y_n + (1 - t_n)\ln(1 - y_n)\right]$  Multi-class: $E(w) = -\sum_n \sum_k t_{nk} \ln y_{nk}$

```python
p = np.clip(y_pred, 1e-15, 1 - 1e-15)
-np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))   # binary
-np.sum(y_true * np.log(p))                                  # multi-class
```

- **Cross-entropy = negative log-likelihood** (MLE recipe, Session 4, slide 2). Binary: each label is a Bernoulli(p) variable, with likelihood `p^t (1−p)^(1−t)`; taking `−log` and summing gives exactly the binary cross-entropy. Multi-class: same with a categorical distribution. So **minimizing the cross-entropy = maximizing the likelihood**, just like MSE = MLE under Gaussian noise for the regressor.
- With one-hot targets, only the term of the **true class** is non-zero: the loss is `−log(probability given to the right answer)`. Giving 0.9 to the right class costs 0.1; giving 0.01 costs 4.6.
- **`np.clip`**: `log(0) = −inf`. If the model predicts exactly 0 for the true class (possible after rounding), the loss would be infinite. Clipping to [1e-15, 1 − 1e-15] avoids it (this is the "underflow in log()" tip of the project, and the same idea as `log(p + 1e-10)` in `ExerciseSolution.ipynb`).
- **Sum, not mean** (like slide 29). See "Why a sum and not a mean?" below.

### Backpropagation

```python
dz2 = y_pred - y                        # (n, k)
dW2 = a1.T @ dz2                        # (h, k)
db2 = dz2.sum(axis=0)                   # (k,)
da1 = dz2 @ self.W2_.T                  # (n, h)
dz1 = da1 * activation_derivative(z1)   # (n, h)
dW1 = X.T @ dz1                         # (d, h)
db1 = dz1.sum(axis=0)                   # (h,)
```

Everything except the first line is **identical to the regressor** (see section 3, line by line).

**Why `dz2 = y_pred − y`?** This is the key result of slide 29. For one sample with sigmoid output `p = σ(z)` and loss `L = −[t ln p + (1−t) ln(1−p)]`:
- `∂L/∂p = −t/p + (1−t)/(1−p) = (p − t) / (p(1−p))`
- `∂p/∂z = σ'(z) = p(1−p)` (section 1)
- chain rule: `∂L/∂z = (p − t)/(p(1−p)) · p(1−p) = p − t`

The `p(1−p)` cancels out. Softmax + categorical cross-entropy gives the same simple result `y_pred − y`. This is why we never need a softmax derivative, and why the sigmoid at the output does **not** cause vanishing gradients (its derivative is cancelled).

**Verification:** analytical gradients compared with finite differences, for binary and 3-class problems: max difference ~1e-9.

### Why a sum and not a mean?
With a mean, the gradient is divided by the batch size, so each step is 32 times smaller. With the tests' settings (`learning_rate=0.001`, 100-200 epochs, about 5 batches per epoch), that is only ~1 000 tiny steps: the classifier stayed near 50 % accuracy. With the **sum** (exactly the formula of slide 29), the same learning rate is enough: all tests pass.
- The **minimum is the same**: multiplying a function by a constant does not move its minimum. Only the effective step size changes (sum = mean with a learning rate × batch size).
- **Loss and gradient use the same convention** (both sums), so `loss_curve_` tracks exactly what is minimized. Because the loss is computed on the whole training set, its value grows with n: divide by n to compare with `MLPClassifier.loss_curve_` (a mean).
- The regressor keeps a mean: its raw targets are of order 100, and a summed gradient would make the steps far too large (divergence).

### `predict_proba`, `predict`, `score`
- **`predict_proba`** returns one column **per class**, like sklearn. In the binary case the network gives only P(class 1), so we add P(class 0) = 1 − p: `np.hstack([1 - y_pred, y_pred])` → `(n, 2)`. The tests check this shape and that each row sums to 1.
- **`predict`**: `self.classes_[np.argmax(proba, axis=1)]`. `argmax` gives the **column** with the highest probability (0, 1, 2…), and `classes_` turns that index back into the original label. In the binary case, this is the same as thresholding P(class 1) at 0.5.
- **`score`**: accuracy = `np.mean(predict(X) == y)`, the share of correct predictions (`True` counts as 1).

### Results (standardized X, 30 hidden neurons, `learning_rate=0.01`, 300 epochs, 30 % test set)

| Dataset | Activation | Ours | `MLPClassifier` |
|---|---|---|---|
| breast cancer (binary) | relu | 0.971 | 0.971 |
| | tanh | 0.977 | 0.977 |
| | logistic | 0.971 | 0.977 |
| iris (3 classes) | relu | 0.911 | 0.844 |
| | tanh | 0.911 | 0.867 |
| | logistic | 0.911 | 0.844 |
| digits (10 classes) | relu | 0.983 | 0.976 |
| | tanh | 0.974 | 0.980 |
| | logistic | 0.972 | 0.976 |

Our model is on par with scikit-learn. On iris it does better, because `MLPClassifier` with Adam and `max_iter=300` has not fully converged (it shows a convergence warning).

### Tests
`pytest -q`: **54/54 pass** (all classifier, regressor, edge-case, smoke and activation tests), including XOR, a single class, 3-class problems and the comparison with sklearn.

---

## 5. Report and results

The report is [`report.ipynb`](report.ipynb). Every number in it comes from a code cell; it runs from top to bottom in about 30 seconds:

```bash
jupyter nbconvert --to notebook --execute --inplace report.ipynb
```

### Protocol (section 0 of the notebook)
- **80/20 train/test split**, `random_state=42`, **stratified** for classification (the test set keeps the class proportions; with 30 iris flowers, an unlucky split could otherwise miss a class almost entirely).
- **`StandardScaler` fitted on the training set only**, then applied to the test set. Fitting it on all the data would leak the test set's mean and variance into training (*data leakage*).
- **Regression targets standardized** with the training mean/std; predictions are transformed back before computing R², so R² is on the original scale.
- Our models: 30 hidden neurons, batches of 32; `learning_rate=0.01` (regressor) and `0.001` (classifier).
- Two scikit-learn references:
  - **sklearn (Adam)**: default `MLPRegressor`/`MLPClassifier` (Adam, L2 `alpha=1e-4`), same hidden size and epoch budget. This is the comparison the project asks for.
  - **sklearn (SGD)**: `solver="sgd"`, `momentum=0`, constant learning rate, `alpha=0`, same batch size, automatic stopping disabled: **the same algorithm as ours**. If our implementation were wrong, this is where it would show.
- **Fair learning rate for the classifier**: our loss is a sum over the batch, sklearn's a mean. A step of 0.001 on a sum of 32 terms equals a step of 0.032 on their mean, so *sklearn (SGD)* gets `learning_rate_init=0.032`.
- **Comparable loss curves**: our regressor's loss `1/(2n) Σ (ŷ − y)²` is exactly sklearn's `squared_error` loss. Our classifier's summed loss is divided by `n_train` to get sklearn's mean cross-entropy.

### Results (test set)

| Model | diabetes (R²) | California (R²) | breast cancer (acc.) | iris (acc.) | digits (acc.) |
|---|---|---|---|---|---|
| ours, relu | 0.474 | 0.742 | 0.965 | 0.933 | 0.978 |
| ours, tanh | 0.471 | 0.722 | 0.947 | 0.933 | 0.969 |
| ours, logistic | 0.469 | 0.673 | 0.965 | 0.933 | 0.961 |
| sklearn (Adam) | 0.495 | 0.746 | 0.956 | 0.867 | 0.978 |
| sklearn (SGD, momentum 0) | 0.480 | 0.737 | 0.947 | 0.933 | 0.975 |
| linear regression | 0.453 | 0.576 | | | |

Epoch budget: 500 (diabetes), 100 (California, 516 mini-batches per epoch), 300 (classification).

- **Our models are on par with scikit-learn** on all five datasets. Against *sklearn (SGD)*, the same algorithm, the gaps are at most 0.02 R² or 1-2 test samples, which come from different random initializations and batch orders.
- *sklearn (Adam)* is slightly ahead on the regression tasks: Adam adapts the step size of every weight. It is behind on iris because it has not converged in 300 epochs (step 0.001, only 4 batches per epoch).
- **Dataset of our choice: California housing** (20 640 districts, 8 features, target = median house value). We chose it because it is 47 times larger than diabetes and non-linear: there, all the networks beat the linear regression by far (0.74 vs 0.58), which shows what the hidden layer brings. On diabetes, the linear regression is only slightly behind (0.45 vs 0.47): the data is small, noisy and mostly linear.
- **One test sample is worth a lot**: 1/114 = 0.9 point on breast cancer, 1/30 = 3.3 points on iris. Differences of one or two samples are not meaningful.

### Analysis (section 5 of the notebook)

**Batch size.**
- *Regressor (mean loss)*, diabetes, relu, `learning_rate=0.01`, 500 epochs:

  | batch_size | updates / epoch | final train loss | test R² |
  |---|---|---|---|
  | None (full) | 1 | 0.229 | 0.479 |
  | 128 | 3 | 0.205 | 0.503 |
  | 32 | 12 | 0.174 | 0.474 |
  | 8 | 45 | 0.134 | 0.289 |

  More updates per epoch means a lower **training** loss, but with batches of 8 (22 500 updates) the network fits the noise of the 353 training patients and the test R² drops: **a lower training loss is not a better model**.
- *Classifier (summed loss)*, breast cancer, relu, `learning_rate=0.1`: the full batch (effective step 45.5 on the mean loss) and batches of 128 (12.8) **diverge** (the loss explodes; the guard stops training and restores the last finite weights), while batches of 32 (3.2) and 8 (0.8) converge. Same learning rate, only the batch size changes: with a summed loss, **the step grows with the batch size**, so the learning rate and the batch size must be chosen together.

**Early stopping**, diabetes, relu, 500 epochs:

| learning_rate | early_stopping | epochs run | train R² | test R² |
|---|---|---|---|---|
| 0.01 | no | 500 | 0.652 | 0.474 |
| 0.01 | yes | 50 | 0.532 | 0.468 |
| 0.1 | no | 500 | 0.639 | **0.127** |
| 0.1 | yes | 15 | 0.538 | **0.477** |

With `learning_rate=0.1` the validation loss reaches its minimum at epoch 5 and goes back up while the training loss keeps decreasing: **overfitting**. Early stopping stops at epoch 15 and restores the epoch-5 weights: test R² 0.127 → 0.477. With `learning_rate=0.01` it mostly saves time (50 epochs instead of 500) for the same test R².

**Standardization.**
- *Raw target* (diabetes, y from 25 to 346): relu **diverges** during the first epoch (caught by the divergence guard, which restores the initial weights: test R² −3.97); tanh and logistic survive but end much lower (test R² 0.236 and 0.355 instead of 0.471 and 0.469). These raw-target runs are numerically unstable, so their exact values can vary slightly from one machine to another. The output error `ŷ − y` starts around 150 instead of around 1, and so does every gradient step.
- *Raw features* (breast cancer, column means from 0.004 to 869): all three activations end up predicting the majority class for every tumour (accuracy 0.632 = share of benign tumours in the test set) instead of 0.95-0.97 with standardized features. Large inputs make `z1` huge: sigmoid and tanh saturate (derivative ≈ 0), relu takes enormous steps.

### What we would say at the oral
- Our implementation **matches scikit-learn's SGD** (same algorithm), which is the best check that the forward pass, the backpropagation and the update rule are correct, on top of the unit tests and the finite-difference gradient check.
- The remaining gap with Adam is the **optimizer**, not the network: Adam (and momentum) are bonus items of the project that we did not implement.
- **Preprocessing** (standardizing X, and y for regression) matters as much as the model: without it the same network saturates or diverges.

---

## 6. Likely oral defense questions

### General
- **Why are there `test_*.py` files?** They are unit tests: they call the functions with inputs whose answer is known and check the output. `pytest` finds and runs every `test_*.py` file. They prove the code is correct and catch regressions when the code is changed later.

### Activations
- **Why use an activation function?** Without it, stacked linear layers collapse into a single linear model. The activation adds non-linearity (e.g. needed for XOR).
- **ReLU vs sigmoid vs tanh in the hidden layer?** ReLU: cheap, no vanishing gradient for z > 0, but can "die". Sigmoid: output in (0, 1), derivative ≤ 0.25 so gradients vanish. Tanh: zero-centered output in (-1, 1), still saturates for large |z|.
- **What is ReLU'(0)?** Not mathematically defined; we use 0 by convention.
- **Why clip in the sigmoid?** To avoid overflow in `np.exp(-z)` for very negative z.
- **Why subtract the max in softmax?** Numerical stability: it prevents overflow and does not change the result, since the constant cancels out between numerator and denominator.
- **Why `keepdims=True`?** To keep shape `(n_samples, 1)` so broadcasting works row by row.
- **Why is there no softmax derivative?** Combined with cross-entropy, the output gradient simplifies to `y_pred - y_true`.

### Base class and training
- **Why not initialize all weights to 0?** Every hidden neuron would compute the same thing and get the same gradient, so they would stay identical forever (symmetry). Random weights break the symmetry.
- **Why Glorot initialization and not std 0.01?** Weights scaled by √(2/(fan_in + fan_out)) keep the variance of activations and gradients stable across layers. With std 0.01, `W2` is so small that the hidden layer gets almost no gradient (plateau): the classifier stayed near 50 % accuracy. Glorot is also what scikit-learn uses.
- **Why `default_rng` and not `np.random.seed`?** It is a local generator: reproducible, and it does not change NumPy's global random state.
- **What is an epoch? A batch?** An epoch is one pass over all training samples. A batch is the subset used for one gradient step. With n = 140 and batch_size = 32, one epoch = 5 updates.
- **Full batch vs mini-batch vs SGD?** Full batch: exact gradient, 1 update per epoch. SGD (batch = 1): very noisy, n updates per epoch. Mini-batch: a compromise, still vectorized inside a batch. Measured here (linear test, sigmoid, 300 epochs): full batch R² = 0.881, batch 32 R² = 0.981. Same cost per epoch, 5 times more updates.
- **Why shuffle every epoch?** So the batches are different every time and the gradient noise is not biased by the order of the data.
- **How does early stopping work, and why restore the best weights?** Hold out a validation set, stop when its loss has not improved by `tol` for `n_iter_no_change` epochs. The last weights are already slightly overfitted, so the best ones are restored.
- **Is there a loop over samples?** No. Only over epochs and batches. Inside a batch, `a1.T @ dz2` does the sum over samples.

### Regressor
- **Why an identity output?** The target is any real number. A sigmoid would bound the output to (0, 1).
- **Why the ½ in the MSE?** It cancels the 2 of the derivative, so the gradient is `(y_pred − y)/n`. Loss and gradient must use the same factor, otherwise `loss_curve_` does not track what we really minimize.
- **Why is `dW2 = a1.T @ dz2`?** Chain rule: `∂L/∂W2[p, j] = Σᵢ a1[i, p] · dz2[i, j]`, and that sum over samples is the matrix product. Shape check: (h, n) @ (n, k) = (h, k) = shape of W2.
- **Why `*` for `dz1` but `@` for `dW1`?** The activation acts element by element (each neuron separately), so the chain rule is element-wise there. Weights mix neurons together, which is a matrix product.
- **How did you check that the gradients are correct?** Finite differences `(L(θ+ε) − L(θ−ε))/2ε` for every parameter: difference ~1e-10.
- **Why reshape y to (n, 1)?** `(n, 1) - (n,)` broadcasts to `(n, n)` silently, which gives a wrong gradient with no error.
- **What does R² < 0 mean?** The model is worse than always predicting the mean of y.
- **Why standardize y on diabetes?** y ≈ 150 makes the output error, and therefore the gradient steps, huge: training diverges. With standardized y, R² ≈ 0.47 (sklearn: 0.50).

### Classifier
- **Why sigmoid for binary and softmax for multi-class?** Sigmoid gives one probability P(class 1), enough for two classes. Softmax gives K probabilities that sum to 1. For K = 2, softmax and sigmoid are equivalent.
- **Why cross-entropy and not MSE?** Cross-entropy is the negative log-likelihood of a Bernoulli/categorical model (MLE). It also gives the simple gradient `y_pred − y`; with MSE + sigmoid, the gradient keeps a factor `σ'(z) ≤ 0.25` and learning stalls when the sigmoid saturates.
- **Why is `dz2 = y_pred − y`?** Chain rule: `∂L/∂p · ∂p/∂z = (p − t)/(p(1−p)) · p(1−p) = p − t`. The derivative of the sigmoid cancels out (slide 29).
- **What is one-hot encoding?** Label k becomes a vector of zeros with a 1 at position k. It matches the K output neurons of the softmax.
- **Why clip in the loss?** `log(0) = −inf`: clipping the probabilities to [1e-15, 1 − 1e-15] keeps the loss finite.
- **Why does `predict_proba` return 2 columns in the binary case?** The sklearn convention: one column per class. We add P(class 0) = 1 − P(class 1).
- **How does `predict` work?** `argmax` of the probabilities gives a column index, and `classes_` maps it back to the original label.
- **Why a summed loss in the classifier but a mean in the regressor?** Same minimum, different step size. The sum (slide 29) learns fast enough with `learning_rate=0.001`. The regressor's raw targets (~100) would make summed steps diverge.
