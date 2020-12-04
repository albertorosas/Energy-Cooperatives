import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


pd.options.display.float_format = "{:,.2f}".format
pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
	if n == -1:
		n = cmap.N
	new_cmap = mcolors.LinearSegmentedColormap.from_list(
		 'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
		 cmap(np.linspace(minval, maxval, n)))
	return new_cmap
minColor = 0.5
maxColor = 0.99
inferno_t = truncate_colormap(plt.get_cmap("coolwarm"), minColor, maxColor)

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

percentage = 50
tail = 8

###################################################################################

#Reading the 3 files
w_proj_now = pd.read_excel(r'C:\Users\alber\Google Drive\0.1 MASTER - SET\pythonProject\LEM2019 BIJLAGE COLLECTIEVE WIND.xlsx', header=6)
w_proj_next = pd.read_excel(r'C:\Users\alber\Google Drive\0.1 MASTER - SET\pythonProject\LEM2019 BIJLAGE COLLECTIEVE WIND.xlsx', 'PIJPLIJN GEPLAND+ VOORBEREIDING', header=6)
s_proj_now = pd.read_excel(r'C:\Users\alber\Google Drive\0.1 MASTER - SET\pythonProject\LEM2019 BIJLAGE COLLECTIEVE ZON.xlsx', header=7)
s_proj_next = pd.read_excel(r'C:\Users\alber\Google Drive\0.1 MASTER - SET\pythonProject\LEM2019 BIJLAGE COLLECTIEVE ZON.xlsx', 'PIJPLIJN GEPLAND+VOORBEREIDING', header=7)
fullcoop = pd.read_excel(r'C:\Users\alber\Google Drive\0.1 MASTER - SET\pythonProject\LEM2019 BIJLAGE ENERGIECOÖPERATIES.xlsx', header=3)

print('#################################################################################################################  COOPERATIVE MOVEMENT ACCOUNTING  \n')
fullcoop = fullcoop.drop(fullcoop.columns[0], axis=1)
muni = fullcoop.GEMEENTE.unique()                                                                                       #This is a list of all the cities where there are cooperatives
coops = fullcoop.NAAM.unique()                                                                                          #This is a list of all cooperative
citynamefix = {'Den Haag': "'s-Gravenhage",
													 'Súdwest-Fryslân':'Sneek',
													 'Het Hogeland': "Eemsmond",
													 'Waadhoeke': "Franekeradeel",
													 'Westerkwartier':'Leek',
													 'Bronckhorst': 'Vorden',
													 "De Fryske Marren": 'Lemsterland',
													 'Midden-Groningen': 'Menterwolde',
													 'Noardeast-Fryslân' : 'Dongeradeel',
													 'De Wolden' : "De Wolden",
													 'West Betuwe': 'Geldermalsen',
													 'Heereveen': "Heerenveen",
													 'Leudal': "Haelen",
													 'Oost Gelre': "Groenlo",
													 'Westerwolde': "Bellingwedde",
													 'Molenlanden':'Giessenlanden',
													 'Montferland':'Didam',
													 'Berkelland':'Borculo',
													 'Emmen':'Emmen',
													 'Gooise Meren':'Naarden',
													 'Stichtse Vecht':'Breukelen',
													 'Rijssen-Holten':'Rijssen',
													 }                                                                                                     #Here the CITY NAME FIX DICTIONARY

def datafix (df):
	df['GEMEENTE'] = df['GEMEENTE'].str.strip().replace(citynamefix)
for i in [fullcoop,s_proj_now,s_proj_next, w_proj_now,w_proj_next]:
	datafix(i)

########################################################################################################################  COUPLE OF PRINTS
print("This is from the COMPLETE cooperative movement: \n", fullcoop, '\n' )
print(f'Acording to such, there are {len(coops)} in {len(muni)} municipalities\n')

########################################################################################################################
bymun = fullcoop.groupby(['GEMEENTE']).size().sort_values(ascending=False)                                              #This tells us how many there are per GEMEENTE
byreg =fullcoop.groupby(['REGIO']).size().sort_values(ascending=False)                                                  #This tells us how many there are per REGIO
byprov =fullcoop.groupby(['PROVINCIE']).size().sort_values(ascending=False)                                             #This tells us how many there are per PROVINCIE
byprov = byprov[:-1]                                                                                                    #Here a litle fix removing last uninteresting value

########################################################################################################################  COUPLE OF PRINTS
print(f'When looking at which are the CITIES with a higher number of coops, we found they are located in \n',
	  fullcoop.groupby(['PROVINCIE', 'GEMEENTE'])['NAAM'].count().sort_values(ascending=False),'\n')                    #This is a paralel way of getting the information out
print(f'When looking at which are the PROVINCES with a higer number of total coops, we found they are located in \n',
	  byprov, '\n')
###################################################################################################################     #This is a query for getting the biggest cooperative and its information
print(f'Zooming into: {bymun.index.values[0]}, the province with mayor number of cooperatives. They are:  \n')
bymun2 = fullcoop.groupby(['GEMEENTE']).get_group(bymun.index.values[0])
print (bymun2.sort_values(by='JAAR', ascending= False), '\n')

print('#################################################################################################################  PER TECHONOLOGY: SOLAR  \n')  #THIS SECTION DEALS ONLY WITH INFORMATION FORM THE EXCEL,

########################################################################################################################  DATA FIXING
def datafix2 (df):
	if 'now' in get_df_name(df):
		df['Source Stamp'] = 'Now'
	if 'next' in get_df_name(df):
		df['Source Stamp'] = 'Next'
	new = df['PROJECTNAAM'].str.split("/", n=1, expand =True)
	try:
		df1 = df[pd.to_numeric(df.loc[:,'VERMOGEN COÖPERATIE (KWP)'], errors='coerce', downcast='float').notnull()]
		df1['VERMOGEN COÖPERATIE (KWP)'] = df1.loc[:,'VERMOGEN COÖPERATIE (KWP)'] * 1.00
		df1= df1.drop(columns=['Unnamed: 0','nr', 'RES-REGIO', 'PROJECTNAAM', 'Kolom1']).sort_values(by='VERMOGEN COÖPERATIE (KWP)', ascending=True)
		df1['REALISATIE JAAR'] = pd.to_datetime(df1['REALISATIE JAAR'], format='%Y')
	except:
		df1 = df[pd.to_numeric(df.loc[:,'VERMOGEN COÖPERATIEF (KW)'], errors='coerce', downcast='float').notnull()]
		df1['VERMOGEN COÖPERATIE (KWP)'] = df1.loc[:,'VERMOGEN COÖPERATIEF (KW)'] * 1.00
		df1= df1.drop(columns=['Unnamed: 0','NR', 'RES-REGIO', 'PROJECTNAAM','VERMOGEN COÖPERATIEF (KW)', 'code']).sort_values(by='VERMOGEN COÖPERATIE (KWP)', ascending=True)
	df1 = df1[df1['VERMOGEN COÖPERATIE (KWP)']>0]
	df1['NAAM']=new[0]
	df1['PROJECT'] = new[1]
	df1 = df1.sort_values(by='VERMOGEN COÖPERATIE (KWP)', ascending=True)
	df1['acum_sum'] = df1.loc[:,'VERMOGEN COÖPERATIE (KWP)'].cumsum()
	df1['cum_per'] = 100 * df1['acum_sum']/df1 ['acum_sum'].max()
	df1= df1.reset_index(drop=True)
	idx =df1.index[df1['cum_per'] >= percentage]
	df1.loc[:idx[0],'GROUP'] = 'SmallType'
	df1.loc[idx[0]:, 'GROUP'] = 'BIGType'
	#project_number_at_percentage = next(x[0] for x in enumerate(s_cumper_now) if x[1] > percentage)
	BIG_COOPS= df1.groupby('NAAM')['VERMOGEN COÖPERATIE (KWP)'].agg(['sum','count']).sort_values(by='sum', ascending=False) #Here solar project are grouped into COOPS, showing total capacity and count per each COOP
	print(df1)
	return df1, BIG_COOPS

def now_plus_next (df_proj_now, df_proj_next):
	df_proj_future = df_proj_now.append(df_proj_next)
	df_proj_future = df_proj_future.sort_values(by='VERMOGEN COÖPERATIE (KWP)', ascending=True)
	df_proj_future['acum_sum2'] = df_proj_future['VERMOGEN COÖPERATIE (KWP)'].cumsum()
	df_proj_future['cum_per2'] = 100 * df_proj_future['acum_sum2']/df_proj_future ['acum_sum2'].max()
	df_proj_future = df_proj_future.reset_index(drop=True)
	bigcoop_future = df_proj_future.groupby('NAAM')['VERMOGEN COÖPERATIE (KWP)'].agg(['sum','count']).sort_values(by='sum', ascending=False)
	print(df_proj_future)
	return df_proj_future

s_proj_now, s_bigcoop_now = datafix2(s_proj_now)
s_proj_next, s_bigcoop_next = datafix2(s_proj_next)
s_proj_future = now_plus_next(s_proj_now, s_proj_next)

w_proj_now, w_bigcoop_now = datafix2(w_proj_now)
w_proj_next, w_bigcoop_next = datafix2(w_proj_next)
w_proj_future = now_plus_next(w_proj_now, w_proj_next)

########################################################################################################################  COUPLE OF PRINTS
print(f'There are {len(s_proj_now.NAAM.unique())} cooperatives doing solar,'
	  f' which together they have {len(s_proj_now)} solar projects'
	  f' in {len(s_proj_now.GEMEENTE.unique())} different cities')                                                       #General State of Affairs for Solar Projects
print('Which, depending on their installation are separated in: \n',
	  s_proj_now.groupby(['DAK/ GROND'])['VERMOGEN COÖPERATIE (KWP)'].sum(), '\n')                                       #Grouped by where are those solar paneles places
########################################################################################################################  FUTURE CASE Breakdown
																														#DONT DELETE, may use
# mask0 = solarcoop.groupby(['NAAM']).get_group(bigsolar.index.values[0])											    #This is a query for the higest in solar project development
# mask1 = solarcoop.groupby(['NAAM']).get_group(bigsolar.index.values[1])                                               #This is a query for the SECOND higest cooperative in solar development
# print ('Ranked by aggregated solar capacity, the biggest cooperatives are: \n', bigsolar, '\n')
# print (f'When looking at the biggest: {bigsolar.index.values[0]}. \n All their project are:\n'
# 	   f'{mask0.sort_values(by=["VERMOGEN COÖPERATIE (KWP)","REALISATIE JAAR"], ascending=False)}'
# 	   f'\n\nTheir finished projects are distribuited across the country in the following manner: \n'
# 	   f'{mask0.groupby(["PROVINCIE", "GEMEENTE"])["VERMOGEN COÖPERATIE (KWP)"].sum().sort_values(ascending=False)}\n\n')
# print (f'When looking at the second biggest: {bigsolar.index.values[1]}. \nAll their project are:\n'
# 	   f'{mask1.sort_values(by=["VERMOGEN COÖPERATIE (KWP)","REALISATIE JAAR"], ascending=False)}'
# 	   f'\n\nTheir finished projects are distribuited across the country in the following manner: \n'
# 	   f'{mask1.groupby(["PROVINCIE", "GEMEENTE"])["VERMOGEN COÖPERATIE (KWP)"].sum().sort_values(ascending=False)}\n\n')
print('#################################################################################################################   Maps are imported: \n')
map_nl0 = gpd.read_file(r'C:\Users\alber\Downloads\gadm36_NLD_shp\gadm36_NLD_0.shp')  #Full country                     #Reading the shapefiles
map_nl1 = gpd.read_file(r'C:\Users\alber\Downloads\gadm36_NLD_shp\gadm36_NLD_1.shp')  #Full PROVINCES
map_nl2 = gpd.read_file(r'C:\Users\alber\Downloads\gadm36_NLD_shp\gadm36_NLD_2.shp')  #Full CITIES
map_nl0 = map_nl0.to_crs(epsg=3395)                                                                                     #Converting into Mercantor Map
map_nl1 = map_nl1.to_crs(epsg=3395)
map_nl2 = map_nl2.to_crs(epsg=3395)
map_nl1 = map_nl1[map_nl1.TYPE_1 != 'Water body']                                                                       #This eliminates the 'Water bodies' from the Provincies map
######################################################################################################################## THE ORIGINALS
#print(map_nl0,'\n')                                                               #This is one big Dutch Polygon
#print(map_nl1,'\n') 															   #This is are the PROVINCIES polygons
print(map_nl2,'\n')
print('#################################################################################################################  Here we do the MERGING into the map. Create an overall "Density view", location specific. This is where our tailored summarries begin\n')

########################################################################################################################  PROVINCIES COOP DISTRIBUITION
for_plotting1 = map_nl1.merge(byprov.rename('Coops'), left_on= 'NAME_1', right_index= True).sort_values(by='Coops', ascending=False)
top1 = for_plotting1.filter(['NAME_1', 'Coops', 'geometry'])
top1 = top1[top1['Coops']>=50]                                                                                          #We care for those with more than 50                        ###SELECTION
top1['Perce'] = (for_plotting1['Coops']/for_plotting1['Coops'].sum())*100                                               #We get their percentage attribution to the overall movement
top1['Perce'] = top1['Perce'].astype(int)
top1['NplusC'] = top1['Coops'].astype(str)+ ' - ' + top1['NAME_1'] + ' '+ top1['Perce'].astype(str) + '%'               #We fix everything into a 'NplusC' Column that show this as we like them on the legend
print (f'Provinces in the Netherlands with more than 50 cooperatives each: \n'
	   f'Together accounting for {top1.Perce.sum()}% of the total movement - 2019 \n', top1, '\n')                      #Resulted worked, USED FOR PLOTTING
########################################################################################################################  GEMEENTE COOP DISTRIBUITION
for_plotting2 = map_nl2.merge(bymun.rename('Coops'), how='inner', left_on= 'NAME_2', right_index= True).sort_values(by='Coops', ascending=False)
top2 = for_plotting2.filter(['NAME_2', 'Coops', 'geometry'])
top2 = top2[top2['Coops']>=4]                                                                                           #We select only GEMEENTES with 4 or more cooperatives       ###SELECTION
top2['Coops'] = top2.Coops.map("{:02}".format)
top2 ['NplusC'] = top2['Coops'].astype(str) + ' - ' + top2['NAME_2']
print(f'GEMEEENTES with 04 or more Cooperatives \n', top2, '\n')                                                        #Resulted work, USE FOR PLOTTING
########################################################################################################################  GEMEENTE SOLAR PROJECT PRESENT DISTRIBUITION

def merging_into_map_solar (s_proj_now):
	s_cities_now = s_proj_now.groupby(['GEMEENTE'])["VERMOGEN COÖPERATIE (KWP)"].sum()
	bysolar2 = s_proj_now.groupby(['GEMEENTE'])["VERMOGEN COÖPERATIE (KWP)"].count().sort_values(ascending=False)        #This is a list of aggregated COUNT of solar projects by city
	for_plotting3 = map_nl2\
		.merge(s_cities_now.rename('Capacity'), left_on='NAME_2', right_index=True)\
		.merge(bysolar2.rename('HowMany'), left_on='NAME_2', right_index=True)\
		.merge(bymun.rename('Coops'), left_on='NAME_2', right_index=True)\
		.sort_values(by='Capacity', ascending=True)                                                                        #We make a big merge
	s_cities_now_geodata = for_plotting3.filter(["NAME_1", 'NAME_2', 'Capacity', "HowMany", "Coops", 'geometry'])

	if 'now' in get_df_name(s_proj_now):
		s_cities_now_geodata['NplusC'] = s_cities_now_geodata.Capacity.map("{:06,.0f}".format) + ' KWP - ' \
										 + s_cities_now_geodata.HowMany.map("{:02d}".format).astype(str) + ' proj. ' \
										 + s_cities_now_geodata['NAME_2']
	if 'future' in get_df_name(s_proj_now):
		s_cities_now_geodata['NplusC'] = s_cities_now_geodata.Capacity.map("{:08,.1f}".format) + ' KWP - ' \
										 + s_cities_now_geodata.HowMany.map("{:02d}".format).astype(str) + ' proj. ' \
										 + s_cities_now_geodata['NAME_2']
	s_cities_now_geodata['acum_sum'] = s_cities_now_geodata['Capacity'].cumsum()
	s_cities_now_geodata['cum_per'] = 100 * s_cities_now_geodata['acum_sum'] / s_cities_now_geodata['acum_sum'].max()

	print(s_cities_now_geodata)
	return s_cities_now_geodata

s_cities_now_geodata = merging_into_map_solar(s_proj_now)
s_cities_future_geodata = merging_into_map_solar(s_proj_future)
w_cities_now_geodata = merging_into_map_solar(w_proj_now)
w_cities_future_geodata = merging_into_map_solar(w_proj_future)


s_acumsum_now = s_proj_now['acum_sum'].to_numpy()
s_acumper_now = s_proj_now['cum_per'].to_numpy()
s_totalcapacity_now = s_proj_now['VERMOGEN COÖPERATIE (KWP)'].sum()                                                           #The Total solar capacity
s_acumsum_fut_numpy = s_proj_future['acum_sum2'].to_numpy()
s_acumper_fut_numpy = s_proj_future['cum_per2'].to_numpy()

w_acumsum_now = w_proj_now['acum_sum'].to_numpy()
w_acumper_now = w_proj_now['cum_per'].to_numpy()
w_acumsum_fut_numpy = w_proj_future['acum_sum2'].to_numpy()
w_acumper_fut_numpy = w_proj_future['cum_per2'].to_numpy()

print('####################################################################################################################################################################### THE PLOTS: \n')

########################################################################################################################  NOW - COOPERATIVE MOVEMENT  #############################################################################################################

#FIGURE  2 MAPS ABOUT COOPERATIVE MOOVEMENT  ###########################################################################  THE MAPS
fc_plot_map, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
#MAP 00
map_nl1.plot(ax=ax, linewidth= 0, edgecolor= 'dimgrey', alpha= 1, color= 'lightgrey')   #This are all the PROVINCES in background grey
top1.plot(ax=ax, column='NplusC',  cmap= 'cividis', legend= True, categorical=True, k=3, label='54654', legend_kwds={'loc': 'lower left'})  #This are the 3 higher PROVINCIES
#map_nl2.geometry.boundary.plot(ax=ax, linewidth=0, edgecolor= 'lightgrey', alpha=.5)   #This are all the division lines of GEMEENTES
leg = ax.get_legend()
leg.set_bbox_to_anchor((0,.8))
ax.set_axis_off()
ax.set(title=f'Provinces in the Netherlands with more than 50 cooperatives each.\n'
			 f'Together accounting for {top1.Perce.sum()}% of the total movement - 2019 ')
#MAP 01
map_nl1.plot(ax=ax1, linewidth= .35, edgecolor= 'grey', alpha= 1, color= 'lightgrey')
#top1.plot(ax=ax1, column='NplusC', cmap= 'cividis', alpha=.35)  #This are the 3 higher PROVINCIES
top2.plot(ax=ax1, column='Coops', cmap= 'Reds', legend = True, categorical=True, k=5, label='bottom',legend_kwds={'loc': 'lower right'})    #This are the 5 higer GEMEENTES
leg1 = ax1.get_legend()
leg1.set_bbox_to_anchor((.2,.65))
ax1.set_axis_off()
ax1.set(title= f'Municipalities with 4 or more cooperatives each. - 2019')
#MAP 02
########################################################################################################################  NOW - TECHONOLOGY  #############################################################################################################

########################################################################################################################  SOLAR   ##############

#FIGURE  MAP GEMMENTES WITH HIGEST CAPACITY   - SOLAR - NOW ############################################################  THE MAP - NOW
s_plot_map_now , ax2 = plt.subplots(figsize=(10, 10))

for_map_solar = s_cities_now_geodata.tail(tail)

map_nl1.plot(ax=ax2, linewidth= .35, edgecolor= 'grey', alpha= 1, color= 'lightgrey')
for_map_solar.plot(ax=ax2, column='NplusC', cmap=inferno_t, legend=True, categorical=True, k=5, legend_kwds={'loc': 'lower right'})
leg2 = ax2.get_legend()
leg2.set_bbox_to_anchor((.3,.65))
ax2.set_axis_off()
ax2.set(title= f'The top {tail} municipalities where the highest cooperative solar capacity is installed.'
			   f' ({for_map_solar.cum_per.max()-for_map_solar.cum_per.min():,.0f}%) ')

#FIGURE  ACCUMULATED CAPACITY OF PROJECTS - SOLAR - NOW ################################################################  THE DISTRIBUITION CURVE  - NOW
s_plot_accum_now, ax = plt.subplots(figsize=(10, 10))

cmap = matplotlib.cm.get_cmap('coolwarm')
rgba1 = cmap(.99)
rgba0 = cmap(0.01)

perce_pos = next(x[0] for x in enumerate(s_acumper_now) if x[1] > percentage)

x = np.linspace(0, len(s_proj_now), len(s_proj_now))
y= s_acumper_now
# 2 SECTIONS LINES
plt.plot(x[:perce_pos], y[:perce_pos], linewidth=3, color= rgba0, label='Distribuited Cooperative Solar')
plt.plot(x[perce_pos:], y[perce_pos:], linewidth=3, color= rgba1, label='Big Project Cooperative Solar')

plt.hlines(percentage, min(s_proj_now['cum_per']), perce_pos, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(perce_pos, 0, percentage, linestyles='dotted', color='grey')
plt.xlim(0)
plt.ylim(0)
ax.set(title=f'From the total {s_totalcapacity_now:,.0f} (KWP) solar capacity installed by cooperatives,\n'
			 f'{percentage}% comes from {perce_pos} projects. '
			 f'Meaning, the remaining {100-percentage}% is attributed to only {(len(s_proj_now) - perce_pos)}')
plt.ylabel("Percentage of total installed capacity (solar) [%]")
plt.xlabel("Number of solar projects")
plt.legend()

#FIGURE  SCATTERED PLOT OF GEMMENTES        - SOLAR -   NOW  ###########################################################  THE SCATTERED PLOT - NOW
s_plot_scate_now, ax = plt.subplots(figsize=(10,10))

perce_pos = next(x[0] for x in enumerate(s_cities_now_geodata.cum_per) if x[1] > percentage)
x=s_cities_now_geodata['HowMany']
y=s_cities_now_geodata['Capacity']
t=x
plt.scatter(x[:perce_pos], y[:perce_pos], color=rgba0, s=s_cities_now_geodata.Coops[:perce_pos] * 25, alpha= .5)
plt.scatter(x[perce_pos:], y[perce_pos:], color=rgba1, s=s_cities_now_geodata.Coops[perce_pos:] * 25, alpha= .5)
plt.xlabel('Number of projects in the city')
plt.ylabel('Aggregated installed capacity (Solar) [KWP]')
ax.set(title='Dots represent cities ')

########################################################################################################################  WIND   ##############

#FIGURE  ACCUMULATED CAPACITY OF PROJECTS - WIND - NOW #################################################################  THE DISTRIBUITION CURVE  - NOW
w_plot_accum_now, ax = plt.subplots(figsize=(10, 10))

cmap = matplotlib.cm.get_cmap('coolwarm')
rgba1 = cmap(.99)
rgba0 = cmap(0.01)

perce_pos = next(x[0] for x in enumerate(w_acumper_now) if x[1] > percentage)

x = np.linspace(0, len(w_proj_now), len(w_proj_now))
y= w_acumper_now
# 2 SECTIONS LINES
plt.plot(x[:perce_pos], y[:perce_pos], linewidth=3, color= rgba0, label='Smaller Project Cooperative Wind')
plt.plot(x[perce_pos:], y[perce_pos:], linewidth=3, color= rgba1, label='Big Project Cooperative Wind')

plt.hlines(percentage, min(w_proj_now['cum_per']), perce_pos, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(perce_pos, 0, percentage, linestyles='dotted', color='grey')
plt.xlim(0)
plt.ylim(0)
ax.set(title=f'From the total {w_acumsum_now.max():,.0f} (KW) wind capacity installed by cooperatives,\n'
			 f'{percentage}% comes from {perce_pos} projects. '
			 f'Meaning, the remaining {100-percentage}% is attributed to only {(len(w_proj_now) - perce_pos)}')
plt.ylabel("Percentage of total installed capacity (wind) [%]")
plt.xlabel("Number of wind projects")
plt.legend()

#FIGURE  MAP GEMMENTES WITH HIGEST CAPACITY   - WIND - NOW ############################################################  THE MAP - NOW
w_plot_map_now , ax2 = plt.subplots(figsize=(10, 10))

for_map_wind = w_cities_now_geodata.tail(tail)

map_nl1.plot(ax=ax2, linewidth= .35, edgecolor= 'grey', alpha= 1, color= 'lightgrey')
for_map_wind.plot(ax=ax2, column='NplusC', cmap=inferno_t, legend=True, categorical=True, k=5, legend_kwds={'loc': 'lower right'})
leg2 = ax2.get_legend()
leg2.set_bbox_to_anchor((.3,.65))
ax2.set_axis_off()
ax2.set(title= f'The top {tail} municipalities where the highest cooperative wind capacity is installed.'
			   f' ({for_map_wind.cum_per.max()-for_map_wind.cum_per.min():,.0f}%) ')

#FIGURE  SCATTERED PLOT OF GEMMENTES        - SOLAR -   NOW  ###########################################################  THE SCATTERED PLOT - NOW
w_plot_scate_now, ax = plt.subplots(figsize=(10,10))

perce_pos = next(x[0] for x in enumerate(w_cities_now_geodata.cum_per) if x[1] > percentage)
x=w_cities_now_geodata['HowMany']
y=w_cities_now_geodata['Capacity']
t=x
plt.scatter(x[:perce_pos], y[:perce_pos], color=rgba0, s=w_cities_now_geodata.Coops[:perce_pos] * 25, alpha= .5)
plt.scatter(x[perce_pos:], y[perce_pos:], color=rgba1, s=w_cities_now_geodata.Coops[perce_pos:] * 25, alpha= .5)
plt.xlabel('Number of projects in the city')
plt.ylabel('Aggregated installed capacity (Solar) [KWP]')
ax.set(title='Dots represent cities ')

########################################################################################################################  FUTURE #############################################################################################################

########################################################################################################################  SOLAR   ##############

#FIGURE  MAP GEMMENTES WITH HIGEST CAPACITY   - SOLAR - FUTURE  ########################################################  THE MAP - FUTURE
s_plot_map_future , ax2 = plt.subplots(figsize=(10, 10))

for_map_solar = s_cities_future_geodata.tail(tail)

map_nl1.plot(ax=ax2, linewidth= .35, edgecolor= 'grey', alpha= 1, color= 'lightgrey')
for_map_solar.plot(ax=ax2, column='NplusC', cmap=inferno_t, legend=True, categorical=True, k=5, legend_kwds={'loc': 'lower right'})
leg2 = ax2.get_legend()
leg2.set_bbox_to_anchor((.3,.65))
ax2.set_axis_off()
ax2.set(title= f'The top {tail} municipalities where the highest cooperative forecasted solar capacity is to be installed.'
			   f' ({for_map_solar.cum_per.max()-for_map_solar.cum_per.min():,.0f}%) ')

#FIGURE  ACCUMULATED CAPACITY OF PROJECTS - SOLAR - FUTURE #############################################################  THE DISTRIBUITION CURVE - FUTURE
perce_pos = next(x[0] for x in enumerate(s_acumper_now) if x[1] > percentage)
project_number_at_percentage2 = next(x[0] for x in enumerate(s_acumper_fut_numpy) if x[1] > percentage)

s_plot_accum_future, ax = plt.subplots(figsize=(10, 10))

x = np.linspace(0, len(s_proj_now), len(s_proj_now))
y= s_acumsum_now

x2 = np.linspace(0,len(s_proj_future),len(s_proj_future))
y2 = s_acumsum_fut_numpy

#plt.plot(x, y, linewidth=3, label= '2019')
plt.plot(x[:perce_pos], y[:perce_pos], linewidth=3, color= rgba0, label='Distribuited Cooperative Solar 2019')
plt.plot(x[perce_pos:], y[perce_pos:], linewidth=3, color= rgba1, label='Big Project Cooperative Solar 2019')

#plt.plot(x2,y2, linewidth= 3, color= 'grey', label= '2019 + Planned Installation' )
plt.plot(x2[:project_number_at_percentage2], y2[:project_number_at_percentage2],linewidth=3, color= rgba0, alpha=.4, label='Distribuited Cooperative Solar 2019 + Planned')
plt.plot(x2[project_number_at_percentage2:], y2[project_number_at_percentage2:],linewidth=3, color= rgba1, alpha=.4, label='Big Project Cooperative Solar 2019 + Planned')

plt.hlines(y[perce_pos], min(s_proj_now['cum_per']), perce_pos, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(perce_pos, 0, y[perce_pos], linestyles='dotted', color='grey')
plt.hlines(y2[project_number_at_percentage2], min(s_proj_now['cum_per']), project_number_at_percentage2, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(project_number_at_percentage2, 0, y2[project_number_at_percentage2], linestyles='dotted', color='grey')

plt.xlim(0)
plt.ylim(0)
ax.set(title=f'From the {s_acumsum_now.max():,.0f} [KWP] solar capacity installed by cooperatives in 2019,\n'
			 f'it is expected to increase up to {s_acumsum_fut_numpy.max():,.0f} [KWP] by 2020\n'
			 f'Meaning, a shift from a ({percentage}% - {100-percentage}%) grouping of '
			 f'{perce_pos} - {(len(s_proj_now) - perce_pos)} projects in 2019'
			 f' to {project_number_at_percentage2} - {len(s_acumper_fut_numpy) - project_number_at_percentage2} in 2020 onwards')
plt.ylabel("Cooperative Solar Capacity [KWP]")
plt.xlabel("Number of solar projects")
plt.legend()

#FIGURE  SCATTERED PLOT OF GEMMENTES        - SOLAR -  FUTURE   ########################################################  THE SCATTERED PLOT - FUTURE
s_plot_scate_future, ax = plt.subplots(figsize=(10,10))

perce_pos = next(x[0] for x in enumerate(s_cities_future_geodata.cum_per) if x[1] > percentage)
x=s_cities_future_geodata['HowMany']
y=s_cities_future_geodata['Capacity']
t=x
plt.scatter(x[:perce_pos], y[:perce_pos], color=rgba0, s=s_cities_future_geodata.Coops[:perce_pos] * 25, alpha= .5)
plt.scatter(x[perce_pos:], y[perce_pos:], color=rgba1, s=s_cities_future_geodata.Coops[perce_pos:] * 25, alpha= .5)
plt.xlabel('Number of projects in the city')
plt.ylabel('Aggregated installed capacity (Solar) [KWP]')
ax.set(title='Dots represent cities ')

########################################################################################################################  WIND   ##############

#FIGURE  ACCUMULATED CAPACITY OF PROJECTS - WIND - FUTURE ##############################################################  THE DISTRIBUITION CURVE  - FUTURE
perce_pos = next(x[0] for x in enumerate(w_acumper_now) if x[1] > percentage)
project_number_at_percentage2 = next(x[0] for x in enumerate(w_acumper_fut_numpy) if x[1] > percentage)

w_plot_accum_future, ax = plt.subplots(figsize=(10, 10))

x = np.linspace(0, len(w_proj_now), len(w_proj_now))
y= w_acumsum_now

x2 = np.linspace(0,len(w_proj_future),len(w_proj_future))
y2 = w_acumsum_fut_numpy

#plt.plot(x, y, linewidth=3, label= '2019')
plt.plot(x[:perce_pos], y[:perce_pos], linewidth=3, color= rgba0, label='Distribuited Cooperative Wind 2019')
plt.plot(x[perce_pos:], y[perce_pos:], linewidth=3, color= rgba1, label='Big Project Cooperative Wind 2019')

#plt.plot(x2,y2, linewidth= 3, color= 'grey', label= '2019 + Planned Installation' )
plt.plot(x2[:project_number_at_percentage2], y2[:project_number_at_percentage2],linewidth=3, color= rgba0, alpha=.4, label='Distribuited Wind 2019 + Planned')
plt.plot(x2[project_number_at_percentage2:], y2[project_number_at_percentage2:],linewidth=3, color= rgba1, alpha=.4, label='Big Project Cooperative Wind 2019 + Planned')

plt.hlines(y[perce_pos], min(w_proj_now['cum_per']), perce_pos, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(perce_pos, 0, y[perce_pos], linestyles='dotted', color='grey')
plt.hlines(y2[project_number_at_percentage2], min(w_proj_now['cum_per']), project_number_at_percentage2, linestyles='dotted', linewidth=.5, color='grey')
plt.vlines(project_number_at_percentage2, 0, y2[project_number_at_percentage2], linestyles='dotted', color='grey')

plt.xlim(0)
plt.ylim(0)
ax.set(title=f'From the {w_acumsum_now.max():,.0f} [KW] solar capacity installed by cooperatives in 2019,\n'
			 f'it is expected to increase up to {w_acumsum_fut_numpy.max():,.0f} [KWP] by 2020\n'
			 f'Meaning, a shift from a ({percentage}% - {100-percentage}%) grouping of '
			 f'{perce_pos} - {(len(w_proj_now) - perce_pos)} projects in 2019'
			 f' to {project_number_at_percentage2} - {len(w_acumper_fut_numpy) - project_number_at_percentage2} in 2020 onwards')
plt.ylabel("Cooperative Wind Capacity [KW]")
plt.xlabel("Number of wind projects")
plt.legend()

#FIGURE  MAP GEMMENTES WITH HIGEST CAPACITY   - WIND - FUTURE  ########################################################  THE MAP - FUTURE
w_plot_map_future , ax2 = plt.subplots(figsize=(10, 10))

for_map_wind = w_cities_future_geodata.tail(tail)

map_nl1.plot(ax=ax2, linewidth= .35, edgecolor= 'grey', alpha= 1, color= 'lightgrey')
for_map_wind.plot(ax=ax2, column='NplusC', cmap=inferno_t, legend=True, categorical=True, k=5, legend_kwds={'loc': 'lower right'})
leg2 = ax2.get_legend()
leg2.set_bbox_to_anchor((.3,.65))
ax2.set_axis_off()
ax2.set(title= f'The top {tail} municipalities where the highest cooperative forecasted solar capacity is to be installed.'
			   f' ({for_map_wind.cum_per.max()-for_map_wind.cum_per.min():,.0f}%) ')

#FIGURE  SCATTERED PLOT OF GEMMENTES        - WIND -  FUTURE   ########################################################  THE SCATTERED PLOT - FUTURE
w_plot_scate_future, ax = plt.subplots(figsize=(10,10))

perce_pos = next(x[0] for x in enumerate(w_cities_future_geodata.cum_per) if x[1] > percentage)
x=w_cities_future_geodata['HowMany']
y=w_cities_future_geodata['Capacity']
t=x
plt.scatter(x[:perce_pos], y[:perce_pos], color=rgba0, s=w_cities_future_geodata.Coops[:perce_pos] * 25, alpha= .5)
plt.scatter(x[perce_pos:], y[perce_pos:], color=rgba1, s=w_cities_future_geodata.Coops[perce_pos:] * 25, alpha= .5)
plt.xlabel('Number of projects in the city')
plt.ylabel('Aggregated installed capacity (Wind) [KW]')
ax.set(title='Dots represent cities ')



plt.close(fc_plot_map)
################################## SOLAR
plt.close(s_plot_map_now)
plt.close(s_plot_accum_now)
plt.close(s_plot_scate_now)
plt.close(s_plot_map_future)
plt.close(s_plot_accum_future)
plt.close(s_plot_scate_future)
################################# WIND
plt.close(w_plot_map_now)
plt.close(w_plot_accum_now)
# plt.close(s_plot_scate_now)
plt.close(w_plot_map_future)
plt.close(w_plot_accum_future)
plt.close(w_plot_scate_future)

#############################################  FUTURE

plt.show()



#FIGURE 02  ############################################################################################################  THE 'FAILED' HISTOGRAM  *** About the PROJECT
# fig2, ax = plt.subplots()
# plt.hist(project_capacity, bins=25)
