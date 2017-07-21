import xlrd

#-------------------------------
def open_file(path):
    """
    Open and read an Excel file
    """
    book = xlrd.open_workbook(path)

    # get the first worksheet
    first_sheet = book.sheet_by_index(0)

    with open('logViewer.csv'.format(first_sheet.name), 'wb') as f:
        writer = csv.writer(f)
        writer = writerows(first_sheet.row_values(row) for row in range(sheet.nrows))


path = "/Users/oscarwlsong/Desktop/testPython/xml.xlsm"
open_file(path)
