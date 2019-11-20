import numpy as np

'''
Byte_min:  5 , Byte_max:  780831
Method_min:  1 , Method_max:  2
Path_min:  0 , Path_max:  14
Ref_min:  2 , Ref_max:  4
Status_min:  200 , Status_max:  502
Time_min:  0 , Time_max:  629
'''


def max_min_scaler(matrix):
    for i in range(len(matrix[0])):
        matrix[0][i] = (matrix[0][i] - 0) / (780831 - 0)
    for i in range(len(matrix[1])):
        matrix[1][i] = (matrix[1][i] - 0) / (2 - 0)
    for i in range(len(matrix[2])):
        matrix[2][i] = (matrix[2][i] - 0) / (14 - 0)
    for i in range(len(matrix[3])):
        matrix[3][i] = (matrix[3][i] - 0) / (4 - 0)
    for i in range(len(matrix[4])):
        matrix[4][i] = (matrix[4][i] - 0) / (502 - 0)
    for i in range(len(matrix[5])):
        matrix[5][i] = (matrix[5][i] - 0) / (629 - 0)
    return matrix
