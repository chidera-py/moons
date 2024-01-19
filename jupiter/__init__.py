#from .moons import Moons
from typing import List
import pandas as pd

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
	
    
	def correlation_matrix(self):
		if self.df is not None:
			corr_matrix = self.df.corr()
			return corr_matrix
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
	
	
