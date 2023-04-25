import csv
import os

class CSVPrinter:
    def __init__(self, file_name):
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name} does not exist.")
        self.file_name = file_name
    def read(self):
        with open(self.file_name) as f:
            reader = csv.reader(f)
            lines = [row for row in reader]
        num_rows = len(lines)
        num_cols = len(lines[0])
        if not all(len(row) == num_cols for row in lines):
            raise ValueError('CSV file contains rows with different number of columns')
        matrix = [[None] * num_cols for _ in range(num_rows)]
        for i in range(num_rows):
            for j in range(num_cols):
                matrix[i][j] = lines[i][j]
        return matrix