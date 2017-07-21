#This function returns the date-weight matrix from the date-area-weight matrix and the required area.
def segAreaTable(DAWmat,area):
    dwmat=[]
    search = area; #assigns the area to search
    for sublist in DAWmat:
        if sublist[1] == search:
            dwmat.append([sublist[0],sublist[2]]) #appends only date and weight to new listt i.e date-weight matrix
    return dwmat

#sample call
DAWmat=[['a', 'ca', 'b'], ['a', 'ba', 'c'], ['b', 'ea', 'd'], ['b', 'ca', 'd']]
dwmat=segAreaTable(DAWmat,'ca')
print(dwmat)