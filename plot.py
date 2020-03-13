import pandas
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# load data
data = pandas.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

# filter cases in Germany
data = data[data['Country/Region'] == 'Germany']
assert(len(data.index) == 1) # check if only one entry for "Germany" found

# remove columns that are not needed anymore
data = data.drop(labels=['Province/State', 'Country/Region', 'Lat', 'Long'], axis='columns')

# check if all remaining columns are valid dates
[parse(entry) for entry in data]

# German data format
data = data.rename(columns={column: parse(column).strftime('%d.%m.%Y') for column in data.columns})

# last date
last_date = data.columns[-1]

# plot
data.iloc[0].plot(grid=True, color='r', title='Fälle Deutschland - Stand ' + last_date)
plt.savefig('cases.png')

data.iloc[0].plot(logy=True, grid=True, color='r', title='Fälle Deutschland (mit logarithmischer Skala) - Stand ' + last_date)
plt.savefig('cases_log.png')
