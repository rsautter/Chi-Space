import pandas as pd
	
def get_SolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/Swind4096norm.csv',header=None).values.T[0]

def get_SurrogateSolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/GaussSW4096.csv',header=None).values.T[0]

def get_endoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zEndo.csv')
	
def get_exoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zExo.csv')

def get_redsDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zRed.csv')
	
def get_LorenzDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zLorenz.csv')

def get_Acoustic1():
	'''
	Source:
		https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/overview
	'''
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/seg_1b1ad8.csv')["acoustic_data"].values

def get_Acoustic2():
	'''
	Source:
		https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/overview
	'''
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/seg_1bd38e.csv')["acoustic_data"].values

def get_WikiTraffic(index=0):
	'''
	The first 100 rows were selected
	
	Source:
		https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/overview
	'''
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/wikipediaWebTraffic.csv').iloc[index].values[1:]



