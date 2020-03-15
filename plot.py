import pandas
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

countries = ['Germany', 'Italy', 'Korea, South', 'France', 'US', 'Japan'] # 'China', 'Spain'
epsilon = 1e-6
translation = {'Germany': 'Deutschland',
        'Italy': 'Italien',
        'Korea, South': 'Südkorea',
        'Spain': 'Spanien',
        'France': 'Frankreich',
        'US': 'USA',
        'Japan': 'Japan',
        'China': 'China'}

# load data
data = pandas.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
# drop provinces of France, data has already accumulated entry
data = data.drop(data[(data['Country/Region'] == 'France') & (data['Province/State'] != 'France')].index)
def collapse(data, country):
    data_country = data[data['Country/Region'] == country].sum()
    data_country['Country/Region'] = country
    data = data.drop(data[data['Country/Region'] == country].index)
    data = data.append(data_country, ignore_index=True)
    return data
# collapse entries of US
data = collapse(data, 'US')
# collapse entries of China
data = collapse(data, 'China')

# filter countries
data = data[data['Country/Region'].isin(countries)]
# check if number of found data rows equals number of queries
assert(len(data.index) == len(countries))
# generate mapping of index to country
indices_countries = {data[data['Country/Region'] == country].index[0]: translation[country] for country in countries}
countries = [translation[country] for country in countries]

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
def restr(a, t):
    # consider growth only
    return np.abs(a), np.abs(t)

def exp_func(x, a, t):
    a, t = restr(a, t)
    return a * np.power(2, x / t) # + c assuming that exp(-inf) = 0

figT, axT = plt.subplots()
index_dates = data.index.to_list()
index = np.arange(len(index_dates))
cases_min = 30 # min cases for fitting curves
xlim_max = 0
for j, country in enumerate(countries):
    cases = np.array(data[country].to_list())
    if j == 0:
        xlim_max = len(cases[cases > cases_min])
    # fit in original space
    (a, t), (a_cov, t_cov) = curve_fit(exp_func, index, cases, maxfev=1000)
    # fit in log space
    mask = cases > cases_min # only consider data with more than 50 cases
    cases_masked = cases[mask]
    index_masked = index[mask]
    b, c = np.polyfit(index_masked, np.log2(cases_masked + epsilon), 1)

    # plot
    fig, ax = plt.subplots()
    ax.plot(index_dates, cases, label='Rohdaten')
    ax.plot(index_dates, exp_func(index, a, t), label='Fit: a=%.3f, t=%.3f' % restr(a, t))
    ax.plot(index_dates, exp_func(index, np.power(2, c), 1 / b), label='Fit (log2): b=%.3f, c=%.3f' % restr(b, c))
    ax.set_title(country)
    ax.set_ylim(-np.max(cases) / 20.0, np.max(cases) * 1.2)
    ax.set_xticks(np.linspace(index[0], index[-1], 6))
    ax.legend()
    fig.savefig(country.replace('ü', 'ue') + '_fit.png')

    # piece-wise fitting of exponential function
    interval = 4 # in days
    fit_over_time = []
    for i in range(len(cases) - interval):
        if cases[i] > cases_min:
            cases_cut = cases[i:i + interval]
            index_cut = index[i:i + interval]
            result = np.polyfit(index_cut, np.log2(cases_cut + epsilon), 1)
            fit_over_time.append(result)
#        else: # do not align start
#            fit_over_time.append([np.nan, np.nan])
    fit_over_time = np.array(fit_over_time)
    axT.plot(range(len(fit_over_time)), 1 / fit_over_time[:, 0], label=country, zorder=-j)

# formatting
axT.legend(loc=2)
axT.set_xlim([0, xlim_max + 5])
axT.set_ylim([-1, 15])
axT.set_xlabel('Tage seit Fälle > ' + str(cases_min))
axT.set_ylabel('Tage bis Fälle verdoppelt')
axT.set_title('Stand ' + last_date)
figT.savefig('tau.png')
