from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

class Moons:
	def __init__(self, data):
		"""
		Initializes the Moons class with a DataFrame loaded from a specified database.
		
		Parameters:
		- data (str): The name of the database containing the moons data.
		
		Returns:
		None
		"""
		
		self.data = data
		
		#Interpreting data base
		database_service = "sqlite"
		
		#Defining where it is stored
		database = f"{self.data}"
		
		connectable = f"{database_service}:///{database}"
		query = "SELECT * FROM moons"
		
		#loads data into a df
		self.df = pd.read_sql(query, connectable, index_col = "moon")
		
        #Adding attributes
		self.columns = self.df.columns
		self.rows = self.df.index
		self.group = self.df.groupby("group")
	
	def load_data(self):
		"""
		Prints out cleaned df
		"""        
		print(self.df)
			 
	def print_groups(self, group_name):
		"""
		Prints specific the groups in the df
		
		Parameters:
		- group_name (str): The name of the group to be printed.
		
		Returns:
		None
		
		"""
		if group_name in self.df["group"].unique():
			for name, group in self.group:
				group = self.group.get_group(group_name)
				if name == group_name:
					print(f"Name: {name} \n")
					print(group)      
		else:
			print(f"Group {group_name} not found")
			 
	#def filter_group(self,"group"):
		# returns data for all moon in a specific group
		#return self.group == group        
	
	def summary(self):
		#returns brief summary of datafields, the num of non null entries and the dtype for each field
		
		if self.df is not None:
			return self.df.info()
		else:
			return "No data loaded. Call .load_data() method first"
		
	def print_row_names(self):
		""" 
		Prints out all the row names in the DF
		
		Returns:
		None
		
		"""
		print(f"{self.rows}")
			 
	def print_col_names(self):
		"""
		Prints out all the column names in the DF
		
		Returns:
		None
		
		"""
		print(f"{self.columns}")
			 
		
	def summary_stats(self):
		""" 
		Returns the summary statistics
		"""
		
		if self.df is not None:
			return self.df.describe()
		else:
			return "No data loaded. Call .load_data() method first"
		
	def correlations(self, var_1:str, var_2:str):
		"""
		Returns correlations between variables
		
		Parameters:
		- var_1 (str): The name of the first variable.
		- var_2 (str): The name of the second variable.
		
		Returns:
		DataFrame: Correlation matrix.
		
		"""
		if self.df is not None:
			if not isinstance(var_1, str) or not isinstance(var_2, str):
				return "Variables must be strings"             
			if var_1 not in self.columns:
				print(f"First variable is not found in moons data frame")
			if var_2 not in self.columns:
				return print(f"Second variable is not found in moons data frame")
			return self.df[[var_1, var_2]].corr()
		else:
			return "No data loaded. Call .load_data() method first"
	
	
	def extract_moons(self, moons: List[str]):
		"""
		extracts data for a specfic moon
		
		Parameters:
		- moons (List[str]): A list of moon names to extract.
		
		Returns:
		DataFrame: Extracted data.
		"""
		
		if self.df is not None:
			if not isinstance(moons, List):
				return "Moons must be entered in a list format"
			for i in range(len(moons)):
				if moons[i] not in self.rows:
					return f"{i+1} moon is not in data frame"
			return self.df.loc[moons]
		else:
			return "No data loaded. Call .load_data() method first"
		
		
	def extract_cols(self, cols: List[str]):
		"""
		extracts data for a specfic coloumn
		
		Parameters:
		- cols (List[str]): A list of column names to extract.
		
		Returns:
		DataFrame: Extracted data.
		"""
		
		if self.df is not None:
			for i in range(len(cols)):
				if cols[i] not in self.columns:
					return f"{i+1} column are not in data frame"
			return self.df[cols]
		else:
			return "No data loaded. Call .load_data() method first"
			 
	def extract_rows(self, rows: List):
		"""
		extracts data for a specific row
		
		Parameters:
		- rows (List[int]): A list of row indices to extract.
		
		Returns:
		DataFrame: Extracted data.
		"""
		
		if self.df is not None:
			for i in range(len(rows)):
				if rows[i] not in self.rows:
					return f" {i+1} row is not in data frame"          
			return self.df.loc[rows]
		else:
			return "No data loaded. Call .load_data() method first"
			 
	def merge(self, cols: List, rows: List):
		"""
		merges data for certain cols and rows
		
		Parameters:
		- cols (List[int]): A list of column indices to merge.
		- rows (List[int]): A list of row indices to merge.
		
		Returns:
		DataFrame: Merged data.
		"""
		if self.df is not None:
			for i in range(len(cols)):
				if cols[i] not in self.columns:
					return f" {i+1} column are not in data frame"
				if rows[i] not in self.rows:
					return f" {i+1} row is not in data frame" 
			return self.df.loc[rows, cols]
		else:
			return "No data loaded. Call .load_data() method first"
	            
	def plot_scatter(self, col_1, col_2):
		"""
		Plots a scatter graph
		
		Parameters:
		- col_1 (str): The name of the first column.
		- col_2 (str): The name of the second column.
		
		Returns:
		None
		"""
		if col_1 and col_2 in self.columns:
			sns.relplot(data = self.df, x = col_1, y = col_2)
			plt.xlabel(col_1)
			plt.ylabel(col_2)
			plt.title(f"{col_1} vs {col_2}")
			plt.show()
		else:
			raise ValueError("Columns are not in df")
			
	def plot_hist(self, col_1):
		"""
		Creates a histogram
		
		Parameters:
		- col_1 (str): The name of the column for the histogram.
		
		Returns:
		None
		"""
		if col_1 in self.columns:
			sns.histplot(data = self.df, x = col_1, bins="auto", kde=True)
			plt.xlabel(col_1)
			plt.ylabel('Frequency')
			plt.title(f'Histogram of {col_1}')
			plt.show()
		else:
			raise ValueError(f"Column {col_1} are not in df")
			
	def plot_box(self, col_1):
		"""
		Generate box plots
		
		Parameters:
		- col_1 (str): The name of the column for box plots.
		
		Returns:
		None
		"""
		if col_1 in self.columns: #checks col_1 is within df
			sns.catplot(data = self.df, y = col_1, x = "group", kind = "box")
		else:
			raise ValueError("Column is not in df")
		    
	def plot_pair(self):
		"""
		Views all pairwise relationships 
		"""
		sns.pairplot(self.df)
		
	def plot_bar(self, col_1):
		'''
		Plots a bar chart for the data against the categorical column 'group' 
		
		Parmeters:
		- col_1 (str)
		
		Raises:
		- ValueError: If 'initial_col' is not found in the DataFrame or 'conversion'is unsupported.
		
		'''
		if col_1 in self.columns: #checks col_1 is within df
			sns.catplot(data = self.df, y = col_1, x = "group", kind = "bar")
			plt.xlabel("Group")
			plt.ylabel(col_1)
			plt.title(f"Bar Chart of {col_1} by Group")
			plt.show()
		else:
			raise ValueError(f"Column {col_1} is not in df")
		
	def convert_units(self, initial_col, new_col_name, conversion):
		"""
		Converts values from one column into a different unit and adds a new column with converted values.
	
		Parameters:
		- initial_col (str): The name of the column containing the original values.
		- new_col_name (str): The name of the new column to be created for the converted values.
		- conversion (str): The type of conversion to be applied. Supports:
		- 'km_to_m': Converts distance from kilometers to meters.
		- 'days_to_secs': Converts time from days to seconds.
	
		Raises:
		- ValueError: If 'initial_col' is not found in the DataFrame or 'conversion' is unsupported.
	
		Updates:
		- Adds a new column with the converted values to the DataFrame.
		- Updates the 'columns' attribute with the new column.
	
		Returns:
		None
		"""
		conversion_factors = {
			'km_to_m': 1000,
			 'days_to_secs': 24*60*60 }  
			 
		if initial_col not in self.columns:
			raise ValueError(f"Initial column '{initial_col}' not found in df.")
		
		if conversion not in conversion_factors:
			raise ValueError(f"Unsupported target unit: '{conversion}'.") 
		
		conversion_factor = conversion_factors[conversion]            
		self.df[new_col_name] = self.df[initial_col] * conversion_factor
			 
		# Update the self.columns attribute with the new column
		self.columns = self.df.columns
		
	def raise_power(self, initial_col:str, new_col_name:str, power:int):
		"""
		Raises an entire column to a specific power and adds a new column with updated values.
		
		Parameters:
		- initial_col (str): The name of the initial column to be raised to the power.
		- new_col_name (str): The name of the new column to be created with the raised values.
	- power (int): The power to which the values in the initial column will be raised.
	
		Raises:
		- ValueError: If the specified initial column is not found in the DataFrame.
	
		Returns:
		None
		"""
		if initial_col not in self.columns:
			raise ValueError("Initial column '{initial_col}' not found in df.")
		
		self.df[new_col_name] = self.df[initial_col] ** power
		
		# Update the self.columns attribute with the new column
		self.columns = self.df.columns
		
	
	
	def linear_reg_model(self, col_1, col_2:List):
		"""
		Performs linear regression analysis on the specified columns, including plotting
		training and testing sets, residuals, and returning relevant coefficients.
	
		Parameters:
		- col_1 (str): Name of the independent variable column (X-axis).
		- col_2 (List): List of dependent variable column names (Y-axis).
	
		Raises:
		- ValueError: If the second variable is not entered in list format or if any of
		the specified columns are not found in the DataFrame.

		Returns:
		Tuple[float, DataFrame, ndarray]:
		- float: The coefficient of the linear regression model.
		- DataFrame: The y_test values used in testing the model.
		- ndarray: The y_pred values predicted by the model.
		"""
		if not isinstance(col_2, List):
			raise ValueError("Second variable must be entered in a list format")
		
		for col in col_2:
			if col not in self.columns:
				raise ValueError(f"{col} is not found in df")
		
		if col_1 not in self.columns:
			raise ValueError(f"{col_1} is not found in df")
		
		# Beginning of linear regression
		X = self.df[[col_1]]
		Y = self.df[col_2]
		
		# Prepare input data, seperating into training and testing sets
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)
		
		# Adding hyperparameters
		model = linear_model.LinearRegression(fit_intercept=True)
		
		#train the model with trainig set
		model.fit(x_train, y_train)
		 
		# use model to predict y-vals of the testing set
		y_pred = model.predict(x_test)
		
		# Create a plot
		fig, ax = plt.subplots()
		
		# Plot for training set
		ax.scatter(x_train, y_train, color='blue', label='Training Set')
		
		# Plot for testing set
		ax.scatter(x_test, y_pred, color='red', label='Testing Set')
		
		# Add axis labels
		ax.set_xlabel(col_1)
		ax.set_ylabel(col_2) 
		
		# Add legend and title
		ax.legend()
		ax.set_title(f"{col_2} Test/Prediction Values against {col_1}")
		
		plt.show()
		
		# Creates a residual plot
		fig,ax = plt.subplots()
		sns.residplot(data = self.df, x = x_test, y = y_test - y_pred)
		
		# Adds a horizontal line at 0
		ax.axhline(0, color = "k", linestyle = "dashed")
		
		# Add axis labels
		ax.set_xlabel(col_1)
		ax.set_ylabel("Residual")
		ax.set_title("Residuals Plot")
		plt.show()
		
		
		# returns the coefficients
		coefficient = model.coef_[0]
		y_intercept = model.intercept_
		
		
		return coefficient, y_test, y_pred
		 
		
	def testing_metrics(self, test, pred):
		"""
		Calculates and prints various regression metrics to evaluate the performance of a model.
	
		Parameters:
		- test : The true values for the testing set.
		- pred : The predicted values generated by the model for  testing set.
	
		Prints:
		- r2_score
		- mean_squared_error: Average squared differences between true and predicted values.
		- root mse: Square root of the mean squared error.
	
		"""
		print(f"r2_score: {r2_score(test, pred)}")
		print(f"mean squared error: {mean_squared_error(test, pred)}")
		print(f"root mse: {mean_squared_error(test, pred, squared = False)}")
        
                
       