# Maths-ML-project

One-hidden-layer neural network (classifier and regressor) implemented from scratch in NumPy, with a scikit-learn-style interface (`fit`, `predict`, `score`). MAT0611 *Mathematics for Machine Learning* course project.

- **Report**: [`report.ipynb`](report.ipynb) (demonstrations on diabetes, breast cancer, California housing, iris and digits; comparison with scikit-learn; loss curves and analysis)
- **Walkthrough of the code and the math**: [`EXPLANATIONS.md`](EXPLANATIONS.md)

```bash
pip install -r requirements.txt
pytest -q                                                       # run the tests
jupyter nbconvert --to notebook --execute --inplace report.ipynb  # re-run the report
```
