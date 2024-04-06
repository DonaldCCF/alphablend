# FactorGene

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

`FactorGene` is a sophisticated Python library that leverages the principles of genetic algorithms, drawing inspiration from `gplearn` and harnessing the power of `scikit-learn`. It is meticulously engineered for the discovery and optimization of predictive factors that possess high information coefficients (IC), making it an essential tool for forecasting market movements, aiming to identify and refine factors that can accurately capture and predict fluctuations within financial markets,

## **Structure**

## **Fitness**

## **Functions**

- Utilizes genetic algorithms for factor discovery and optimization.
- Designed to integrate seamlessly into data processing and machine learning workflows with a `scikit-learn`-like interface.
- Enables the discovery of predictive trading factors to capture stock movements.


## **Dependencies**

- pandas
- scikit-learn

## **Usage**
*Note*: The following example uses `yfinance` to fetch financial market data, which is **not** a direct dependency of `FactorGene`. If you wish to run the example as is, please ensure you have `yfinance` installed:

```bash
pip install yfinance
```

```
import pandas as pd
from factorgene import GeneticOptimizer, function_map
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

# Prepare the dataset for FactorGene
y = (features_dict['Open'].shift(-2) / features_dict['Open'].shift(-1) - 1).dropna(how='all')
X = {key: df.loc[y.index] for key, df in features_dict.items()}

# Initialize and fit the Genetic Optimizer
optimizer = GeneticOptimizer(generations=10, population_size=100, function_set=list(function_map.keys()), verbose=1)
optimizer.fit(X, y)

# Display the best programs found
print(optimizer.best_programs_)
```

## **Reference**


The development and theoretical foundation of `FactorGene` are inspired and informed by significant works in the field of genetic algorithms and their application to financial modeling. Below are key references that have influenced this project:

- Jansen, Stefan. "Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python, 2nd Edition." Packt Publishing, 2020.
- Kakushadze, Zura. "101 Formulaic Alphas" (December 9, 2015). Wilmott Magazine, issue 84 (2016), pages 72-80.
- Koza, John R. "Genetic Programming: On the Programming of Computers by Means of Natural Selection." MIT Press, 1992. 
- Poli, Riccardo, et al. "A Field Guide to Genetic Programming." Lulu.com, 2008.
  
These references have provided valuable insights and methodologies that have been instrumental in the development of `FactorGene`.
