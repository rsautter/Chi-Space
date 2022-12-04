import pandas as pd
	
def get_SolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/Swind4096norm.csv',header=None).values.T

def get_SurrogateSolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/GaussSW4096.csv',header=None).values.T

def get_endoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zEndo.csv')
	
def get_exoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zExo.csv')

def get_redsDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zRed.csv')
	
def get_LorenzDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zLorenz.csv')
