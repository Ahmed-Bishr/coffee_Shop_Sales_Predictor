# Coffee Shop — Linear Regression Demo

## Project Overview
This small project demonstrates a simple linear regression implemented from scratch in a Jupyter notebook (`coffeShop.ipynb`). The goal is to model how the number of cups sold changes with temperature using ordinary least-squares solved via gradient descent.

## What this repository contains
- `coffeShop.ipynb` — Jupyter notebook with the full implementation and plotting.
- `README.md` — this file (project overview, setup, usage, and notes).
- `LICENSE` — project license (MIT by default).

## Technical summary
The notebook implements a single-feature linear model:

y_hat = w * x + b

Where:
- `x` is temperature (°C)
- `y` is sold cups
- `w` is the weight (slope)
- `b` is the bias (intercept)

Gradient descent is used to minimize the mean squared error (MSE) cost:

J(w,b) = 1/(2m) * sum((y_hat_i - y_i)^2)

Gradients used for parameter updates are calculated across the training examples.

## How to run locally
1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install required packages:

```bash
pip install numpy matplotlib jupyterlab
```

(If you use the notebook kernel that was configured in the workspace, the kernel likely already contains the required packages.)

3. Open the notebook:

```bash
jupyter lab coffeShop.ipynb
# or
jupyter notebook coffeShop.ipynb
```

4. Run the notebook cells in order. The notebook computes model parameters and shows two plots: cost history and the fitted line.

## Files of interest (notebook cell numbers)
- Cell 1: imports and setup
- Cell 2: dataset definition
- Cell 3: `linearRegression` function
- Cells 4–6: loss, gradient, gradient descent
- Final cells: run `main_app()` and plot

## Notes & suggestions
- The implementation is intentionally explicit for educational purposes (no scikit-learn). For production or larger datasets, prefer NumPy vectorized operations or use libraries like scikit-learn.
- You can reduce `iterations` or increase `alpha` in `main_app()` if training takes too long.

## Contributing
If you want enhancements (vectorized implementation, train/test split, evaluation metrics, or exporting model parameters), open an issue or PR.

## License
This project is licensed under the MIT License — see `LICENSE`.

---

If you'd like a different license (Apache-2.0, GPL) or changes to the README wording/format, tell me and I'll update it.