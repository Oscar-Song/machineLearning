import matplotlib.pyplot as plt
import numpy as np

'''
Class Name: Final Computation
Purpose: A final computation module specifically for outputting graphs for each functional area in the ACLTOP instrument
Functions: __init__(), loadData(), createGraph()
Author: Sunjay Ravishankar
'''
class FinalComputation(object):

    '''
    Function Name: __init__()
    Purpose: Default function which initializes the threshold dictionary for ACLTOP with given weights.
    It also initializes lists that will hold x_axis and y_axis values for the respective functional areas.
    It creates a dictionary to hold all of the lists initalized and loaded with data corresponding to each functional area.
    '''
    def __init__(self):
        self.threshold_dict = {'ARMs': 0.80, 'Barcode/Racks': 0.607, 'Cuvettes Movement': 0.705, 'LLD (Liquid Level Detection)': 0.700,
        'ORU': 0.700, 'Others': 0.600, 'SW': 0.400, 'Fluidics': 0.400}
        self.x_axis = []
        self.y_axis = []
        self.y_axis_ARMs = []
        self.y_axis_Cuvettes = []
        self.y_axis_BarcodeRack = []
        self.y_axis_LLD = []
        self.y_axis_ORU = []
        self.y_axis_Others = []
        self.y_axis_SW = []
        self.y_axis_Fluidics = []
        self.x_labels = []

        self.data_dict = dict()

	'''
	Function Name: loadData()
	Input: z_scores - a matrix containing the data in the first column, the area in the second column,
	and the z_score for that day in the third column.
	Purpose: loads all of the list attributes with data for graphing that the FinalComputation() object has.
	Should be called once prior to calling createGraph()

	'''
    def loadData(self, z_scores):
        #loop through each row and add data to its respective functional area
        for row in z_scores:
            self.x_axis.append(row[1])
            if row[0] == 'ARMs':
                self.y_axis_ARMs.append(row[2])
            elif row[0] == 'Barcode/Racks':
                self.y_axis_BarcodeRack.append(row[2])
            elif row[0] == 'Cuvettes Movement':
                self.y_axis_Cuvettes.append(row[2])
            elif row[0] == 'LLD (Liquid Level Detection)':
                self.y_axis_LLD.append(row[2])
            elif row[0] == 'ORU':
                self.y_axis_ORU.append(row[2])
            elif row[0] == 'Others':
                self.y_axis_Others.append(row[2])
            elif row[0] == 'SW':
                self.y_axis_SW.append(row[2])
            elif row[0] == 'Fluidics':
                self.y_axis_Fluidics.append(row[2])

            #create a list of labels for the x_axis of the output graph
            self.x_labels = sorted(list(set(self.x_axis)))

            #load the dictionary with the lists corresponding to each functional area
            self.data_dict['ARMs'] = self.y_axis_ARMs
            self.data_dict['Barcode/Racks'] = self.y_axis_BarcodeRack
            self.data_dict['Cuvettes Movement'] = self.y_axis_Cuvettes
            self.data_dict['LLD (Liquid Level Detection)'] = self.y_axis_LLD
            self.data_dict['ORU'] = self.y_axis_ORU
            self.data_dict['Others'] = self.y_axis_Others
            self.data_dict['SW'] = self.y_axis_SW
            self.data_dict['Fluidics'] = self.y_axis_Fluidics


	'''
	Function Name: createGraph()
	Input: area - a string with the functional area to graph
	serial_number - a string with the serial number of the machine from which the
	area being graphed comes from
	Purpose: A function to graph the specific functional area being queried. Saves the
	figure to the current directory. Must be called after loadData has been called.
	'''

    def createGraph(self, area, serial_number):
        #retrieve the data for the specified functional area

        working_list = self.data_dict[area]


        xs = [i for i in range(len(working_list))]
        f = plt.figure(area, figsize=(8.00,6.00), dpi=100)
        ax = f.add_subplot(111)

        #label the graph
        plt.ylabel('Z-Score')
        plt.title('Line Graph of Z-Score ' + area + ' ' + serial_number)

        #plot the data
        ax.plot(xs, working_list,'blue', xs, working_list, 'bo')
        threshold = [self.threshold_dict[area] for i in range(len(xs))]
        plt.ylim(0,2.5)

        #label the ticks
        plt.xticks(xs,self.x_labels,rotation=35)
        xt = ax.xaxis.get_ticklabels()
        for i in range(len(xt)):
            xt[i].set_fontsize(10)
            xt[i].set_horizontalalignment("right")
            if i % 2 == 0:
                xt[i].set_visible(False)
        plt.plot(xs,threshold, 'red')

        #save the figure
        if area == 'Barcode/Racks':
            f.savefig(serial_number + '_BarcodeRacks.png', dpi=100)

        else:
            f.savefig(serial_number + '_' + area + '.png', dpi=100)
