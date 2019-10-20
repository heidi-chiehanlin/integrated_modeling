from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import matplotlib
import warnings

class ConsoleSettings(): 
    def __init__(self): 
        pass
    
    def set_all(self): 
        self.set_autoreload()
        self.set_output_all_results()
        self.set_pandas_output_format()
        self.set_matplotlib_output_format()
        self.set_ignore_warnings()
        
    def set_output_all_results(self): 
        """
        Output Setting: print all outcomes in cell (default: only prints the last outcome)
        """
        
        InteractiveShell.ast_node_interactivity = "all"
    
    def set_pandas_output_format(self): 
        """
        Output Setting: set pandas output format 
        """
        
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 200)
        pd.set_option("precision", 4)

    def set_matplotlib_output_format(self): 
        """
        Output Setting: matplotlib
        """
        
        matplotlib.matplotlib_fname()
        
    def set_ignore_warnings(self):
        """
        Output Setting: ignore warninings 
        """
        
        warnings.simplefilter('ignore')
        
    def set_autoreload(self, set_ = 0): 
        command = "%autoreload {}".format(set_)
        InteractiveShell.exec_lines = [command]
        
