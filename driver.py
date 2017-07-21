'''
This is the main module of the MachineLearning package
It calls and coordinate all the other supporting python
files.
Author: Oscar Song, Sunjay Ravishankqr, Janelle Lines, Oshin Mundada
Input: logbook.xlsm
Output: TBD
'''

# Import libraries and other supporting modules
from sys import argv
from dataFilter import DataFilter
from ExcelToTsv import Converter

script, xfile = argv    # xfile is the excel file

# Declare objects to convert excel file into csv file and filter out the data
converter = Converter(xfile)
filter = DataFilter()

tsv = converter.convert()
matrix3DRep =  filter.filterTable(tsv)
print matrix3DRep
#matrix3D = filter.sumDay(matrix3DRep)


'''
areas = ['ARMs', 'Barcode/Racks', 'Cuvettes Movement', 'Fluidics', 'LLD (Liquid Level Detection)', 'ORU', 'Others', 'SW', 'Temperature']

for area in areas:
    matrix2D = filter.segAreaTable(matrix3D, area)
    print area
    print matrix2D
'''
