# Project Name: Titanic Survival Prediction

This project is a data science exploration of the Titanic dataset, focusing on predicting the survival of passengers aboard the RMS Titanic. The dataset contains information about passengers' demographics, such as age, gender, and class, as well as details about their ticket fare, cabin, and embarkation point.

The primary goal of this project is to analyze the Titanic dataset and identify patterns that may have influenced passengers' chances of survival. By applying various machine learning techniques and feature engineering strategies, we aim to build a predictive model with high accuracy in determining whether a passenger survived or not.

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.x
- virtualenv
- pip

## Setting Up the Environment

1. **Clone the repository** and navigate to the project folder
2. Create a virtual environment using `virtualenv`:
```
virtualenv venv
```

3. **Activate the virtual environment**:

- **Windows:**

```
venv\Scripts\activate
```

- **macOS/Linux:**

```
source venv/bin/activate
```

4. **Install the required dependencies** from `requirements.txt`:
```
pip install -r requirements.txt
```
## Running the model
To run the model you need to simply run the `model.py` file after activating your virtua environment

```
cd titanic_nb
python model.py
```



## Running Unit Tests

To run the unit tests, use the `pytest` command along with the `coverage` tool:

```
coverage run -m pytest
```

This command will discover and execute all tests in the project, while also tracking code coverage.

## Check Coverage Results
To view the code coverage results, use the `coverage report` command:

```
coverage report
```

This will display a summary of the code coverage for each module in the project.
