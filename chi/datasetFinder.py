import pandas as pd
	
def get_SolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/Swind4096norm.csv',header=None).values.T[0]

def get_SurrogateSolarWind():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/GaussSW4096.csv',header=None).values.T[0]
	
def get_SYMH():
	'''
	The reference paper is: 
	https://www.researchgate.net/publication/342285814_Disentangling_nonlinear_geomagnetic_variability_during_magnetic_storms_and_quiescence_by_timescale_dependent_recurrence_properties
	The data source available at: https://omniweb.gsfc.nasa.gov/form/omni_min.html
	'''
	return pd.read_csv(r"https://raw.githubusercontent.com/rsautter/Chi-Space/main/data/SYM.H.csv")

def get_endoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zEndo.csv')
	
def get_exoDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zExo.csv')

def get_redsDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zRed.csv')
	
def get_LorenzDataResults():
	return pd.read_csv(r'https://raw.githubusercontent.com/rsautter/Chi-Space/main/results/zLorenz.csv')
	
def get_TemperatureGISS():
	return pd.read_csv("https://data.giss.nasa.gov/gistemp/graphs_v4/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.csv",skiprows=1)["No_Smoothing"]
