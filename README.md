# AlphaGene

<!--ts-->
* [Introduction](#introduction)
* [Structure](#structure)
* [Fitness](#fitness)
* [Functions](#functions)
* [Dependencies](#dependencies)
* [Usage](#usage)
* [Reference](#reference)
<!--te-->

## **Introduction**

`AlphaGene` is a sophisticated Python library that leverages the principles of genetic algorithms, drawing inspiration from `gplearn` and harnessing the power of `scikit-learn`. It is meticulously engineered for the discovery and optimization of predictive alphas that possess high information coefficients (IC), making it an essential tool for forecasting market movements, aiming to identify and refine alphas that can accurately capture and predict fluctuations within financial markets,

Genetic programming represents a robust methodology for prompting computers to automatically address a problem based solely on a high-level description of the desired outcome (Koza, 1992). It is a domain-independent approach that evolves a population of computer programs to resolve a given issue. Genetic programming systematically modifies a group of computer programs generation after generation by applying mechanisms analogous to natural genetic processes.

The evolutionary operations employed in genetic programming include:
- **Crossover**
- **Mutation**
- **Reproduction**
- **Gene duplication**
- **Gene deletion**

Occasionally, analogs of developmental stages are used to morph an embryonic form into a mature structure.

![Genetic Programming Main Loop](https://github.com/donaldccf/AlphaGene/assets/117000928/68a92d75-41ed-4e88-9f94-35a161bdd820)<br />
*Figure 1: Illustration of the main loop of genetic programming (Koza, 1992).*

Unlike genetic algorithms where the population structures are fixed-length character strings that symbolize potential solutions, genetic programming deals directly with programs that act as candidate solutions upon execution.

### Program Representation

In genetic programming, programs are usually represented as syntax trees rather than lines of code. For instance, the expression <br /> `max(x * x, x + 3 * y)` is visualized through a tree structure that highlights nodes and links—the nodes represent instructions, while the links define the arguments for each instruction. The inner nodes of such a tree are termed functions, and the tree's leaves are referred to as terminals.<br /> 
<img src="https://github.com/donaldccf/AlphaGene/assets/117000928/d8676292-c583-41cd-8ee7-d56b11f48c7c" width="250" alt="AlphaGene Diagram"> <br /> 
*Figure 2: Basic tree-like program representation used in genetic programming (Koza, 1992).*

## **Structure**

### 1. Input Data
- **X**: A dictionary of DataFrames, which can include various types of data such as open-high-low-close (OHLC) values, momentum indicators, growth metrics, and other financial alphas.
- **y**: A DataFrame that represents the target variable. This could be data that we aim to predict or find the highest correlation with, such as the return on the next day.

### 2. Genetic Algorithm
The input data, consisting of the matrices from **X** and the target variable **y**, are processed through a genetic algorithm. Within the `AlphaGene` framework, this algorithm seeks to identify and enhance predictive alphas. It employs evolutionary techniques like mutation, crossover, and selection to iteratively improve the predictive power of these alphas.

### 3. Output
The culmination of the genetic algorithm's evolutionary process is a collection of optimized alphas. These alphas are anticipated to exhibit high Information Coefficients (IC), indicating their predictive strength. The final output is a set of programs, each representing a formula that combines various alphas. Accompanying each program is its corresponding fitness score, which quantifies its effectiveness in capturing the relationship with the target variable.

<img src="https://github.com/donaldccf/AlphaGene/assets/117000928/808d4f05-1bb4-47a5-9287-b2d5aae3b54a" width="650" alt="AlphaGene Diagram">   <br /> 
*Figure 3: Structure of AlphaGene.*

## **Fitness**

In the `AlphaGene` library, fitness is a crucial metric used to evaluate the performance of a program. It quantifies how well the program's predicted values (`y_pred`) align with the actual target values (`y`). The fitness score is computed using specific functions that measure the correlation between `y_pred` and `y`.

### IC (Information Coefficient)
- **Type**: Cross-sectional Pearson correlation.
- **Description**: The IC fitness function calculates the Pearson correlation coefficient across different cross-sections. It measures the linear relationship between the predicted values and the actual target values. A higher Pearson correlation indicates a stronger linear relationship, with a score closer to 1 signifying a perfect positive linear correlation and a score closer to -1 indicating a perfect negative linear correlation.

### RIC (Rank Information Coefficient)
- **Type**: Cross-sectional Spearman correlation.
- **Description**: The RIC fitness function computes the weighted Spearman correlation coefficient, which is a non-parametric measure of rank correlation. It assesses how well the relationship between the predicted values and the actual target values can be described using a monotonic function. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed and is based on the ranked values rather than the raw data.


## **Functions**

The `AlphaGene` library includes a variety of functions that are essential for financial time-series analysis. These functions can be categorized into standard mathematical operations, operators, and specialized functions derived from financial analysis concepts, many of which are inspired by WorldQuant LLC's research.

### Standard Mathematical Functions and Operators
- **abs(x)**: Absolute value of `x`.
- **log(x)**: Natural logarithm of `x`.
- **sign(x)**: Sign function, indicating whether `x` is positive, negative, or zero.
- **Operators**: Standard arithmetic (`+`, `-`, `*`, `/`) are implemented with their usual definitions.

### Specialized Financial Functions
- **rank(x)**: Computes the cross-sectional rank of `x`.
- **delay(x, d)**: Retrieves the value of `x` from `d` days ago.
- **correlation(x, y, d)**: Calculates the time-series correlation of `x` and `y` over the past `d` days.
- **covariance(x, y, d)**: Computes the time-series covariance of `x` and `y` over the past `d` days.
- **scale(x, a)**: Rescales `x` such that the sum of the absolute values equals `a` (default is `a = 1`).
- **delta(x, d)**: The difference between today's value of `x` and the value of `x` from `d` days ago.
- **signedpower(x, a)**: Raises `x` to the power of `a`, preserving the sign of `x`.
- **decay_linear(x, d)**: Applies a weighted moving average on `x` over the past `d` days with weights linearly decreasing from `d` to `1`.

### Time-Series Functions
- **ts_min(x, d)**: Minimum of `x` over the past `d` days.
- **ts_max(x, d)**: Maximum of `x` over the past `d` days.
- **ts_argmin(x, d)**: Day on which the minimum of `x` occurred over the past `d` days.
- **ts_argmax(x, d)**: Day on which the maximum of `x` occurred over the past `d` days.
- **ts_rank(x, d)**: Rank of `x` over the past `d` days.
- **sum(x, d)**: Sum of `x` over the past `d` days.
- **product(x, d)**: Product of `x` over the past `d` days.
- **stddev(x, d)**: Standard deviation of `x` over the past `d` days.

The integers used in these functions represent different time horizons, capturing short-, medium-, and long-term trends or patterns in the financial data. This allows for a nuanced analysis that can adapt to various investment strategies and timeframes.


## **Dependencies**

- pandas
- scikit-learn

## **Usage**
*Note*: The following example uses `yfinance` to fetch financial market data, which is **not** a direct dependency of `AlphaGene`. If you wish to run the example as is, please ensure you have `yfinance` installed:

```bash
pip install yfinance
```

```
import pandas as pd
from alphagene import GeneticOptimizer, function_map
import yfinance as yf

# Load stock symbols
stocks = pd.read_csv('stock_list.csv', index_col=0).index[:100]

# Fetch and prepare data using yfinance
features_dict = {'Open': pd.DataFrame(),
                 'High': pd.DataFrame(),
                 'Low': pd.DataFrame(),
                 'Close': pd.DataFrame(),
                 'Volume': pd.DataFrame()}

for stock in stocks:
    data = yf.download(stock, start='2018-01-01')
    for category in features_dict:
        features_dict[category][stock] = data[category]

# Prepare the dataset for AlphaGene
y = (features_dict['Open'].shift(-2) / features_dict['Open'].shift(-1) - 1).dropna(how='all')
X = {key: df.loc[y.index] for key, df in features_dict.items()}

# Initialize and fit the Genetic Optimizer
optimizer = GeneticOptimizer(generations=10, population_size=100, function_set=list(function_map.keys()), verbose=1)
optimizer.fit(X, y)

# Display the best programs found
print(optimizer.best_programs_)
```
<img src="https://github.com/donaldccf/AlphaGene/assets/117000928/c4fca44d-54d6-4903-81c3-deca66948c1a" width="400" alt="AlphaGene Diagram"> <br /> 
*Figure 4: Top 10 Alpha genrated from the data.*

<img src="https://github.com/donaldccf/AlphaGene/assets/117000928/cfecce19-c0b7-49a8-84af-22c5b911d189" width="150" alt="AlphaGene Diagram"> <br />
*Figure 5: Example of the program.*

## **Reference**


The development and theoretical foundation of `AlphaGene` are inspired and informed by significant works in the field of genetic algorithms and their application to financial modeling. Below are key references that have influenced this project:

- Jansen, Stefan. "Machine Learning for Algorithmic Trading, 2nd Edition." Packt Publishing, 2020.
- Kakushadze, Zura. "101 Formulaic Alphas" (December 9, 2015). Wilmott Magazine, issue 84 (2016), pages 72-80.
- Koza, John R. "Genetic Programming." MIT Press, 1992. 
- Poli, Riccardo, et al. "A Field Guide to Genetic Programming." Lulu.com, 2008.
  
These references have provided valuable insights and methodologies that have been instrumental in the development of `AlphaGene`.
