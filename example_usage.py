import pandas as pd
from factorgene import GeneticOptimizer, make_function, function_map
import time
import yfinance as yf

stocks = pd.read_csv('stock_list.csv', index_col=0).index[:100]

# Dates for which you want the data
start_date = '2018-01-01'

# Initialize empty DataFrames for OHLC
features_dict = {
    'Open': pd.DataFrame(),
    'High': pd.DataFrame(),
    'Low': pd.DataFrame(),
    'Close': pd.DataFrame(),
    'Volume': pd.DataFrame(),
}

for stock in stocks:
    data = yf.download(stock, start=start_date)
    for category in features_dict.keys():
        features_dict[category][stock] = data[category]

y = (features_dict['Open'].shift(-2) / features_dict['Open'].shift(-1) - 1).dropna(how='all')
X = {key: df.loc[y.index] for key, df in features_dict.items()}

function_set = list(function_map.keys())

self = GeneticOptimizer(generations=10, population_size=1000,
                        hall_of_fame=100, n_components=100,
                        init_depth=(2, 6),
                        function_set=function_set,
                        parsimony_coefficient=0.00005,
                        max_samples=1, verbose=1,
                        random_state=int(time.time()), n_jobs=-1, )
self.fit(X, y)
best_programs = self._best_programs
data = [{'Equation': str(program), 'Fitness': program.fitness_} for program in best_programs]
df_programs = pd.DataFrame(data).drop_duplicates()
df_programs = df_programs.sort_values(by='Fitness', ascending=False, ignore_index=True)

# best_programs[0].execute(X)
# print(best_programs[0].export_graphviz())
