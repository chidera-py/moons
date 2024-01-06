class Moons:
	def __init__(self, data):
		self.data = data
		
		#initalise an empty df
		self.df = None
	
	def load_data(self):
		#Interpreting data base
		database_service = "sqlite"
		
		#Defining where it is stored
		database = "{self.data}"
		
		connectable = f"{database_service}:///{database}"
       		query = "SELECT * FROM moons"
		
		#loads data into a df
		self.df = pd.read_sql(query, connectable)
		
	
	def summary(self):
		#returns brief summary of datafields, the num of non null entries and the dtype for each field
		
		if self.df is not None:
			return self.df.info()	
		
	def summary_stats(self):
		#returns the summary statistics
		
		if self.df is not None:
			return self.df.describe()
		else:
			return "No data loaded. call load_data first"
		
	def correlations(self, var_1, var_2):
		#returns correlations between variables
		
		if self.df is not None:
			return self.df[[var_1, var_2]].corr()
	
	def correlation_matrix(self, var_1: str, var_2: str):
		if self.df is not None:
			corr_matrix= self.df[[var_1][var_2]].corr()
			return corr_matrix
	
	def extract_moon(self, moon):
		# extracts data for a specfic moon
		
		if self.df is not None:
			return self.df[['moon'] == moon ]

	def extract_cols(self, cols: List[str]):
		# extracts data for a specfic coloumn

		if self.df is not None:
			return self.df[cols]
	
	def extract_row(self, rows: List[int]):
		# extracts data for a specific row
		
		if self.df is not None:
			return self.df.loc[rows

	def merge(self, cols: List[int], rows: List[int]):

		# merges data for certain cols and rows
		
		if self.df is not None:
			return self.df.loc[rows, cols]
	
	
