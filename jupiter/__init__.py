from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
	def __init__(self, data):
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
		# Prints out cleaned df        
		print(self.df)
     
	def print_groups(self, group_name):
		#Prints specific the groups in the df
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
		#Prints out all the row names in the DF
		print(f"{self.rows}")
        
	def print_col_names(self):
		#Prints out all the column names in the DF
		print(f"{self.columns}")
              
	def summary_stats(self):
		#returns the summary statistics
		
		if self.df is not None:
			return self.df.describe()
		else:
			return "No data loaded. Call .load_data() method first"
		
	def correlations(self, var_1:str, var_2:str):
		#returns correlations between variables
		
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
		# extracts data for a specfic moon
		
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
		# extracts data for a specfic coloumn

		if self.df is not None:
			for i in range(len(cols)):
				if cols[i] not in self.columns:
					return f"{i+1} column are not in data frame"
			return self.df[cols]
		else:
			return "No data loaded. Call .load_data() method first"
        
	def extract_rows(self, rows: List[int]):
		# extracts data for a specific row
		
		if self.df is not None:
			for i in range(len(rows)):
				if rows[i] not in self.rows:
					return f" {i+1} row is not in data frame"          
			return self.df.loc[rows]
		else:
			return "No data loaded. Call .load_data() method first"
        
	def merge(self, cols: List[int], rows: List[int]):
		# merges data for certain cols and rows
		
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
		if col_1 and col_2 in self.columns:
			sns.relplot(data = self.df, x = col_1, y = col_2)
			plt.xlabel(col_1)
			plt.ylabel(col_2)
			plt.title(f"{col_1} vs {col_2}")
			plt.show()
		else:
			raise ValueError("Columns are not in df")
			
	def plot_hist(self, col_1):
		#create a histogram 
		if col_1 in self.columns:
			sns.histplot(data = self.df, x = col_1, bins="auto", kde=True)
			plt.xlabel(col_1)
			plt.ylabel('Frequency')
			plt.title(f'Histogram of {col_1}')
			plt.show()
		else:
			raise ValueError(f"Column {col_1} are not in df")
			
	def plot_box(self, col_1):
		#generate box plots 
		if col_1 in self.columns: #checks col_1 is within df
			sns.catplot(data = self.df, y = col_1, x = "group", kind = "box")
		else:
			raise ValueError("Column is not in df")
		    
	def plot_pair(self):
		#View of all pairwise relationships
		sns.pairplot(self.df)
		
	def plot_bar(self, col_1):
		#Makes a bar plot       
		if col_1 in self.columns: #checks col_1 is within df
			sns.catplot(data = self.df, y = col_1, x = "group", kind = "bar")
			plt.xlabel("Group")
			plt.ylabel(col_1)
			plt.title(f"Bar Chart of {col_1} by Group")
			plt.show()
		else:
			raise ValueError(f"Column {col_1} is not in df")
		
	def convert_units(self, initial_col, new_col_name, new_units):
		#converts vals from one col into a different unit and adds new column with converted values
        
		conversion_factors = {
			'kilometers_to_metres': 1000,
			' }      
		if intital_col not in self.columns:
			raise ValueError(f"Initial column '{initial_col}' not found in Df.")
            
		if new_units not in conversion_factors:
			raise ValueError(f"Unsupported target unit: '{new_units}'.") 
            
		conversion_factor = conversion_factors[new_units]            
		self.df[new_col_name] = self.df[intial_col] * conversion_factor
            
		
	def raise_power(self, initial_col, new_col_name, power:int):
		#raises an entire col to a specified power and adds new colum with raised values
		if initial_col not in self.columns:
			raise ValueError("Initial column '{initial_col}' not found in Df.")
            
		self.df[new_col_name] = self.df[intital_col] ** power
    
    
	#def training(self, ):
		
	#def testng(self, ):
		
	#def predict(self, ):
		
	#def linear_regression(self, ):
        
        
        
                
       