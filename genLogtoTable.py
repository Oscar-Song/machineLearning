import numpy as np
import xlrd
import glob
import zipfile

'''
Program Name: genLogtoTable.py
Purpose: Class that takes in a list of generallog input parameters and creates a matrix containing
information about the error code, error type, date, macro area, and weight of the error
Author: Sunjay Ravishankar
'''
class GeneralLog(object):
    def __init__(self):
        self.final_list_of_matrices = []

	'''
	Function Name: error_message_dictionary
	Purpose: Reads the Error Code List .xlsx file and creates a dictionary with the macro area
	and weight for the corresponding error code
	output: returns a matrix dictionary containing error codes, macro area, and weight
	'''
    def error_message_dictionary(self):
        #initialize the error dictionary
        error_dict = {}

        #open the workbook containing all of the error messages and get the sheet
        error_workbook = xlrd.open_workbook('Error Code List - ACL TOP Logbook Viewer Rev 3 with Pre-Analytical Warning and Error Codes.xlsx')
        error_sheet = error_workbook.sheet_by_index(0)

        #loop through the rows skipping the header
        for row in xrange(1, error_sheet.nrows):

            #column counter for accessing the cells
            col_counter = 1

            #loop through each cell in the row
            for cell in error_sheet.row_values(row):

                #if in the first column get the error code information
                if col_counter == 1:
                    error_code = str('%05d' % int(cell))
                elif col_counter == 4: #get the macro area information
                    if cell == '':
                        macro_area = 'Others'
                    else:
						macro_area = str(cell)
                elif col_counter == 9: #get the error_weight information
                    if cell == '':
                        weight_index = 0.0
                    else:
                        weight_index = float(cell)

                col_counter += 1

            #assign code with macro area and weight index
            error_dict[error_code] = [macro_area, weight_index]

        #return the dictionary
        return error_dict

	'''
	Function Name: create_matrix()
	Purpose: Uses the error code dictionary to create a matrix with error code, date, error type,
	macro area, and weight for use in the autoencoder
	Input: a list of general log text files to read
	Output: returns a n x 5 matrix
	'''
    def create_matrix(self, text_file_list):
        self.text_file_list = text_file_list

        # get the error dictionary
        e_dict = self.error_message_dictionary()


        #loop through all of the text files in the list
        for text_file in self.text_file_list:
            temp_matrix = []

            #open the current text file and parse it
            with open(text_file, 'r') as input_file:
                file_lines = input_file.read().split('\n')

                #delete the header line
                del file_lines[0]

                del file_lines[len(file_lines)-1]
                #del file_lines[len(file_lines)-1]

                #loop through each line in the text file
                for line in file_lines:
                    temp_matrix_row = []
                    line = line.split('\t')

                    #get the necessary information
                    code = line[0]
                    if len(code)==0:
                        continue
                    if code[0] == "\'":
                        code = code[1:]


                    #avoid codes that are missing in the dictionary
                    if code not in e_dict.keys():
                        continue

                    date = line[2][:10]

                    mac_area = e_dict[code][0]
                    weight = e_dict[code][1]

                    #append each field to a row that will be added to the matrix
                    temp_matrix_row.append(date)
                    temp_matrix_row.append(mac_area)
                    temp_matrix_row.append(weight)

                    temp_matrix.append(temp_matrix_row)

                self.final_list_of_matrices.append(temp_matrix)

        return self.final_list_of_matrices

        '''
        Function Name: create_one_matrix()
        Purpose: Uses the error code dictionary to create a matrix with error code, date, error type,
        macro area, and weight for use in the autoencoder
        Input: a single general log text files to read
        Output: returns a n x 5 matrix
        '''
    def create_one_matrix(self, text_file):
        self.text_file = text_file

        # get the error dictionary
        e_dict = self.error_message_dictionary()
        t_matrix = []

        #loop through all of the text files in the list

        #open the current text file and parse it
        with open(text_file, 'r') as input_file:
            file_lines = input_file.read().split('\n')

            #delete the header line
            del file_lines[0]

            del file_lines[len(file_lines)-1]
            #del file_lines[len(file_lines)-1]

            #loop through each line in the text file
            for line in file_lines:
                t_matrix_row = []
                line = line.split('\t')

                #get the necessary information
                code = line[0]
                if len(code) == 0:
                    continue
                if code[0] == "\'":
                    code = code[1:]


                #avoid codes that are missing in the dictionary
                if code not in e_dict.keys():
                    continue

                date = line[2][:10]

                mac_area = e_dict[code][0]
                weight = e_dict[code][1]

                #append each field to a row that will be added to the matrix
                t_matrix_row.append(date)
                t_matrix_row.append(mac_area)
                t_matrix_row.append(weight)

                t_matrix.append(t_matrix_row)

            return t_matrix

	'''
	Function Name: get_all_files()
	Purpose: Finds all the .utf8.txt files in the specified sub-folder. These files
	are all of the generalLog.txt files properly encoded for use in training the
	neural network
	Output: returns a list of all the files with extension .utf8.txt
	'''
    def get_all_files(self, directory, folder_name ):
        path = str(directory + folder_name + '/*.utf8.txt')
        my_files = glob.glob(path)

        list_of_files = []

        for a_file in my_files:
            list_of_files.append(a_file)

        return list_of_files


    '''
    Purpose: Just like get_all_files() but reads a zip file
    Author: Oscar Song, Janelle Lines

    '''
    def get_zip_files(self,directory,folder_name):
        fh = open(folder_name, 'rb')
        z = zipfile.ZipFile(fh)


        list_of_files = []
        for name in z.namelist():
            if name.startswith('utf8 100/'):
                print name
                outpath = directory
                #list_of_files.append(z.extract(name,outpath))

        fh.close()
        return z.namelist()
