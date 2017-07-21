import xlrd

#-------------------------------
def open_file(path):
    """
    Open and read an Excel file
    """
    book = xlrd.open_workbook(path)

    #print number of sheets
    print book.nsheets 
    
    #print sheet names
    print book.sheet_names()

    # get the first worksheet
    first_sheet = book.sheet_by_index(0)
    
    #read the sheet
    num_cols = first_sheet.ncols    # number of columns
    for row in range(5, first_sheet.nrows): #iterate through rows
        for col in range(0, num_cols):      #iterate through columns
            cell_obj = first_sheet.cell(row, col) #get cell object by row, col
            print cell_obj,
            print " ",    
        print ""


path = "/Users/oscarwlsong/Desktop/testPython/xml.xlsm"
open_file(path)
