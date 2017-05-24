import numpy as np
from pySTATIS.data_storage import STATISData

X1_name = 'Sub-01'
X1 = np.array([[8, 6, 7, 4, 1, 6],
      [7, 5, 8, 1, 2, 8],
      [6, 5, 6, 5, 3, 4],
      [9, 6, 8, 4, 3, 5],
      [2, 2, 2, 8, 7, 3],
      [3, 4, 4, 9, 6, 1],
      [5, 3, 5, 4, 8, 3],
      [5, 2, 4, 8, 7, 4],
      [8, 6, 8, 4, 4, 7],
      [4, 6, 2, 5, 3, 4],
      [8, 4, 8, 1, 3, 3],
      [5, 3, 6, 4, 4, 2]])
X1_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Smoky', 'Citrus'],
X1_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X2_name = 'Sub-02'
X2 = np.array([[8, 6, 8, 3, 7, 5],
      [6, 5, 6, 3, 7, 7],
      [6, 6, 6, 5, 8, 7],
      [8, 6, 8, 4, 6, 6],
      [2, 3, 1, 7, 4, 3],
      [4, 3, 4, 9, 3, 5],
      [3, 3, 2, 7, 4, 4],
      [4, 3, 5, 5, 3, 3],
      [8, 6, 9, 5, 5, 6],
      [5, 5, 5, 6, 5, 8],
      [8, 4, 8, 3, 7, 7],
      [5, 3, 7, 4, 8, 5]])
X2_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Tropical', 'Leafy'],
X2_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X3_name = 'Sub-03'
X3 = np.array([[8, 6, 8, 3, 7, 2],
      [8, 7, 7, 2, 8, 2],
      [8, 7, 7, 6, 9, 1],
      [8, 2, 8, 3, 9, 3],
      [3, 4, 3, 6, 4, 6],
      [4, 3, 4, 8, 3, 9],
      [5, 4, 5, 2, 3, 6],
      [6, 3, 7, 7, 1, 7],
      [8, 5, 9, 1, 5, 2],
      [5, 5, 4, 6, 5, 1],
      [8, 3, 7, 3, 5, 4],
      [5, 4, 4, 5, 4, 3]])
X3_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Grassy', 'Flinty']
X3_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X4_name = 'Sub-04'
X4 = np.array([[9, 5, 8, 2, 6],
      [8, 7, 7, 3, 5],
      [8, 8, 9, 2, 7],
      [8, 8, 9, 4, 7],
      [4, 2, 2, 4, 3],
      [3, 2, 2, 6, 2],
      [4, 4, 4, 6, 4],
      [5, 2, 2, 9, 4],
      [7, 5, 6, 3, 2],
      [5, 6, 6, 4, 4],
      [7, 3, 6, 1, 6],
      [5, 2, 2, 6, 6]])
X4_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Leafy']
X4_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X5_name = 'Sub-05'
X5 = np.array([[9, 6, 9, 3, 8, 2],
      [7, 7, 7, 1, 9, 2],
      [7, 7, 7, 1, 7, 2],
      [8, 9, 7, 5, 6, 1],
      [4, 4, 4, 2, 4, 4],
      [4, 5, 5, 6, 1, 5],
      [6, 5, 7, 2, 3, 1],
      [6, 6, 5, 8, 4, 5],
      [8, 6, 8, 2, 5, 4],
      [6, 6, 6, 4, 6, 3],
      [7, 4, 8, 4, 5, 1],
      [5, 5, 5, 5, 6, 1]])
X5_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Vegetal', 'Hay']
X5_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X6_name = 'Sub-06'
X6 = np.array([[8, 5, 6, 2, 9],
      [6, 6, 6, 2, 4],
      [7, 7, 7, 2, 7],
      [8, 7, 8, 2, 8],
      [3, 2, 2, 7, 2],
      [3, 3, 3, 3, 4],
      [4, 2, 3, 3, 3],
      [5, 3, 5, 9, 3],
      [7, 7, 7, 1, 4],
      [4, 6, 2, 4, 6],
      [7, 4, 8, 2, 3],
      [4, 5, 3, 3, 7]])
X6_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Melon']
X6_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X7_name = 'Sub-07'
X7 = np.array([[8, 5, 8, 4],
      [7, 6, 8, 4],
      [6, 7, 6, 3],
      [7, 8, 6, 1],
      [4, 2, 3, 6],
      [4, 4, 4, 4],
      [4, 3, 4, 4],
      [5, 3, 5, 7],
      [8, 4, 9, 4],
      [4, 7, 5, 2],
      [8, 5, 7, 3],
      [4, 3, 5, 2]])
X7_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral']
X7_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X8_name = 'Sub-08'
X8 = np.array([[7, 6, 7, 4, 9, 2],
      [6, 5, 6, 2, 7, 2],
      [6, 6, 6, 4, 9, 2],
      [8, 7, 8, 2, 8, 2],
      [3, 3, 4, 4, 4, 4],
      [4, 4, 4, 7, 3, 6],
      [5, 3, 5, 3, 3, 5],
      [6, 4, 6, 3, 2, 4],
      [8, 6, 5, 4, 5, 4],
      [5, 7, 5, 4, 6, 1],
      [7, 4, 8, 2, 6, 2],
      [5, 4, 6, 2, 4, 3]])
X8_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Cutgrass', 'Smoky']
X8_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X9_name = 'Sub-09'
X9 = np.array([[8, 6, 9, 1, 7],
      [8, 7, 9, 1, 6],
      [7, 7, 8, 4, 7],
      [8, 9, 9, 3, 9],
      [3, 4, 4, 5, 4],
      [5, 5, 5, 7, 2],
      [5, 5, 5, 6, 3],
      [5, 5, 6, 5, 3],
      [8, 7, 8, 4, 7],
      [5, 6, 4, 5, 6],
      [8, 4, 7, 4, 5],
      [5, 4, 5, 3, 4]])
X9_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral', 'Peach']
X9_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']

X10_name = 'Sub-10'
X10 = np.array([[8, 6, 7, 5],
       [7, 5, 7, 3],
       [7, 6, 6, 2],
       [8, 7, 7, 4],
       [2, 3, 1, 7],
       [3, 3, 3, 9],
       [4, 2, 5, 8],
       [3, 4, 2, 8],
       [8, 6, 7, 4],
       [5, 6, 4, 4],
       [7, 4, 8, 5],
       [5, 4, 6, 6]])

X10_col_names = ['CatPee', 'PassionFruit', 'GreenPepper', 'Mineral']
X10_row_names = ['NZ_1', 'NZ_2', 'NZ_3', 'NZ_4', 'FR_1', 'FR_2', 'FR_3', 'FR_4', 'CA_1', 'CA_2', 'CA_3', 'CA_4']


def get_wine_data():
    X = [STATISData(X = X1, ID = X1_name, normalize = ('zscore', 'norm_one'), col_names = X1_col_names, row_names = X1_row_names),
         STATISData(X = X2, ID = X2_name, normalize = ('zscore', 'norm_one'), col_names = X2_col_names, row_names = X1_row_names),
         STATISData(X = X3, ID = X3_name, normalize = ('zscore', 'norm_one'), col_names = X3_col_names, row_names = X1_row_names),
         STATISData(X = X4, ID = X4_name, normalize = ('zscore', 'norm_one'), col_names = X4_col_names, row_names = X1_row_names),
         STATISData(X = X5, ID = X5_name, normalize = ('zscore', 'norm_one'), col_names = X5_col_names, row_names = X1_row_names),
         STATISData(X = X6, ID = X6_name, normalize = ('zscore', 'norm_one'), col_names = X6_col_names, row_names = X1_row_names),
         STATISData(X = X7, ID = X7_name, normalize = ('zscore', 'norm_one'), col_names = X7_col_names, row_names = X1_row_names),
         STATISData(X = X8, ID = X8_name, normalize = ('zscore', 'norm_one'), col_names = X8_col_names, row_names = X1_row_names),
         STATISData(X = X9, ID = X9_name, normalize = ('zscore', 'norm_one'), col_names = X9_col_names, row_names = X1_row_names),
         STATISData(X = X10, ID = X10_name, normalize = ('zscore', 'norm_one'), col_names = X10_col_names, row_names = X1_row_names)
         ]

    return X
