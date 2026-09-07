# Machine-learning-projects-del-dia-uno
building mini to moderate projects while figuring out as I learn every Machine Learning topics, sub-topics, concepts and algorithms.


-----
Project topic: HOUSE PRICE PREDICTOR 

INFO ABOUT HOUSE PRICE PREDICTOR AS FOLLOWS: -

###1. It's a linear Regression model, the whole project has been built by me on a synthetic dataset inspired by sklearn's California housing features, generated using Numpy.

###2. Concepts Covered: 
    - Feature engineering/extracting, feature framing out of the California house dataset by Data         frames (via Pandas)
    - Train-Test-Split (via Scikit-learn/ sklearn.model_selection)
    - Coefficients and Intercept (via sklearn.linear_model)
    - Gradient descent - adjusting numbers with weights, biases, noise (via sklearn)
    - Model Evaluation - MSE, MAE, R^2 (Scikit-learn/sklearn.metrics) 
    - Visualization - actual prices of the house vs model's predicted prices (via Matplotlib)


###3. Results: ** MSE: 0.30 | R^2: 0.98 | MAE: 0.44

----

###4. Tech Stack Used: -
      - Python (the whole training has been done on Python programming language)
      - Numpy (numerical python library)
      - Pandas (python library for Feature Engineering)
      - Scikit-Learn (Machine Learning tool of python)
      - Matplotlib (python tool for making plots through visualisation)
      - Google Colab - .ipynb file (the IDE where this whole training code has been executed)

# Gradient Descent from Scratch 📉 with the help of linear regression algorithm

Built using **only NumPy** — no sklearn, no black box.

## What this project does
Implements linear regression by building gradient descent 
from scratch, without using any ML libraries.
The model learns the relationship between house sizes and 
prices purely through math and iteration.

## Concepts covered
- What gradient descent actually is under the hood
- How weights (w) and bias (b) update every iteration
- What loss/cost function means and why it decreases
- Why sklearn's LinearRegression().fit() works the way it does

## How it works
Start with random weights (w=0, b=0)
↓
Make a prediction: y_pred = w * X + b
↓
Measure how wrong it is (MSE loss)
↓
Calculate gradients (which direction to adjust)
↓
Update weights: w = w - learning_rate * gradient
↓
Repeat 1000 times → model learns


## Results
| | Value |
|---|---|
| True w | 3.0 |
| Learned w | 2.79 |
| True b | 4.0 |
| Learned b | 4.18 |
| Final Loss | 0.807 |

Model reached **93% accuracy** on true parameters 
using pure math.

## What I learned
Most people use sklearn's LinearRegression() as a black box.
Building this from scratch showed me exactly what happens 
inside that one line of code — the same fundamental process 
that trains every ML model in the world, including 
large language models like ChatGPT and Claude.

## Tech stack
- Python 3.14
- NumPy (math only — no ML libraries)
- Matplotlib (visualization)


