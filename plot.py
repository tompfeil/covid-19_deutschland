import pandas
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
