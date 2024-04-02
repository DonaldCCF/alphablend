import random
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from alphablend import FeatureBlender, make_function
import yfinance as yf
import time

def delay1(df1):
    return df1.shift(1)

def rank(df1):
    return df1.rank(axis=1)

def correlation(df1, df2):
    pass

delay_1 = make_function(function=delay1, name='delay', arity=1)
rank1 = make_function(function=rank, name='rank', arity=1)

stocks = pd.read_csv('stock_list.csv', index_col=0).index[:100]

# Dates for which you want the data
start_date = '2022-01-01'

# Initialize empty DataFrames for OHLC
features_dict = {
    'Open': pd.read_csv('data/open.csv', index_col=0, parse_dates=True),
    'High':  pd.read_csv('data/high.csv', index_col=0, parse_dates=True),
    'Low':  pd.read_csv('data/low.csv', index_col=0, parse_dates=True),
    'Close':  pd.read_csv('data/close.csv', index_col=0, parse_dates=True),
    'Volume':  pd.read_csv('data/volume.csv', index_col=0, parse_dates=True)
}

# for stock in stocks:
#     data = yf.download(stock, start=start_date)
#     for category in features_dict.keys():
#         features_dict[category][stock] = data[category]

y = (features_dict['Open'].shift(-2) / features_dict['Open'].shift(-1) - 1).dropna(how='all')
X = {key: df.loc[y.index] for key, df in features_dict.items()}

function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min', delay_1]

self = FeatureBlender(generations=10, population_size=10000,
                      hall_of_fame=100, n_components=100,
                      init_depth=(2, 6),
                      function_set=function_set,
                      parsimony_coefficient=0.0005,
                      max_samples=0.9, verbose=1,
                      random_state=int(time.time()), n_jobs=-1,)
self.fit(X, y)
best_programs = self._best_programs
data = [{'Equation': str(program), 'Fitness': program.fitness_} for program in best_programs]
df_programs = pd.DataFrame(data).drop_duplicates()
df_programs = df_programs.sort_values(by='Fitness', ascending=False, ignore_index=True)

# best_programs[0].execute(X)
# print(best_programs[0].export_graphviz())
