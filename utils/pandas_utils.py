#!/usr/bin/env python3

import pandas as pd
import os

class PandasUtils():
    def __init__(self):
        pass
    
    def readCsvFile(self, file_path, index):
        csv_file = pd.read_csv(file_path, names=index, header=None)
        return csv_file

    def findColor(self, color_name, color_file_path=os.getcwd()+'/utils/colors/basic_colors.csv', index=["color","color_name","hex","R","G","B"]):
        csv_data = self.readCsvFile(file_path=color_file_path, index=index)
        row = self.filterDataFrameByColumn(csv_data, 'color_name', color_name)
        index = self.getIndexesOfDataFrame(row)
        r = int(self.getCellValueByColumnName(row, index[0], 'R'))
        g = int(self.getCellValueByColumnName(row, index[0], 'G'))
        b = int(self.getCellValueByColumnName(row, index[0], 'B'))
        return (b, g, r)

    def filterDataFrameByColumn(self, df, column_name, parameter_name):
        # Select Rows where Column is Equal to Specific Value
        return df.loc[df[column_name] == parameter_name]

    def getCellValueByColumnName(self, df, index, column_name):
        return df.loc[index][column_name]
    
    def getIndexesOfDataFrame(self, df):
        return (list(df.index.values.tolist()))