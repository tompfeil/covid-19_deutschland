import pandas
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

countries = ['Germany', 'Italy', 'Korea, South']

# load data
data = pandas.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

# filter cases in Germany
data = data[data['Country/Region'].isin(countries)]
# generate mapping of index to country
indices_countries = {index: countries[i] for i, index in enumerate(data.index.to_list())}
# check if number of found data rows equals number of queries
assert(len(data.index) == len(countries))

# remove columns that are not needed anymore
data = data.drop(labels=['Province/State', 'Country/Region', 'Lat', 'Long'], axis='columns')
# check if all remaining columns are valid dates
[parse(entry) for entry in data]
# check if no day is missing
for day_before, day in zip(data.columns, data.columns[1:]):
    assert((parse(day) - parse(day_before)).total_seconds() == 60 * 60 * 24)

# German data format
data = data.rename(columns={column: parse(column).strftime('%d.%m.%Y') for column in data.columns})
# last date
last_date = data.columns[-1]

# dates should be rows, not columns
data = data.transpose()
# replace index by name
data = data.rename(columns=indices_countries)
# plot
data.plot(grid=True, title='Fälle - Stand ' + last_date)
plt.savefig('cases.png')
# log plot
data.plot(logy=True, grid=True, title='Fälle (auf logarithmischer Skala) - Stand ' + last_date)
plt.savefig('cases_log.png')

# fitting exponential functions to data
def restr(a, t, c):
    # consider growth only
    return np.abs(a), np.abs(t), c

def exp_func(x, a, t, c):
    a, t, c = restr(a, t, c)
    return a * np.exp(x / t) + c

for country in countries:
    cases = data[country].to_list()
    index = range(len(cases))
    fit = curve_fit(exp_func, index, cases, maxfev=1000)

    index_dates = data.index.to_list()
    fig, ax = plt.subplots()
    ax.plot(index_dates, cases, label='data')
    ax.plot(index_dates, exp_func(index, *fit[0]), label='fit a=%f, t=%f, c=%f' % tuple(restr(*fit[0])))
    ax.set_title(country)
    ax.set_ylim(-np.max(cases) / 20.0, np.max(cases) * 1.2)
    ax.set_xticks(np.linspace(index[0], index[-1], 6))
    ax.legend()
    fig.savefig(country + '_fit.png')
