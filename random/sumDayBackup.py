    def sumDay(self, matrix):
        newMatrix = []
        newMatrix.append(matrix[0])

        for i in range(1, len(matrix)):
            if matrix[i][DATE] == matrix[i-1][DATE] and matrix[i][MACRO_AREA] == matrix[i-1][MACRO_AREA]:

                newMatrix[len(newMatrix)-1][WEIGHT] += matrix[i][WEIGHT]
            else:
                newMatrix.append(matrix[i])
        return newMatrix
