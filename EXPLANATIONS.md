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

*To be completed.*

---

## 3. `slp_regressor.py`

*To be completed.*

---

## 4. `slp_classifier.py`

*To be completed.*

---

## 5. Report and results

*To be completed.*

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
