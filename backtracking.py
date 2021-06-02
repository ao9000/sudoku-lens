import numpy as np

BLANK_STATE = 0


def backtracking(board):
    index = get_next_index(board)

    # Check if index was found
    if index:
        # Unpack index
        row_index, column_index = index

        # Try 1-9
        for num in range(1, 10):
            # Check if current loop number is valid for the cell
            if check_valid(board, (row_index, column_index), num):
                # Valid, assign number to cell
                board[row_index][column_index] = num

                # Move on to next available cell
                board = backtracking(board)

                # Check if puzzle is solved
                if not get_next_index(board):
                    return board

        board[row_index][column_index] = 0
        return board
    else:
        # No empty cell, meaning puzzle has been completed
        return board


def get_next_index(board):
    for row_index in range(0, 9):
        for column_index in range(0, 9):
            if board[row_index][column_index] == BLANK_STATE:
                return row_index, column_index

    return None


def check_valid(board, index, cell_value):
    # Unpack index
    row_index, column_index = index

    # Check if number is valid
    # Check row
    if cell_value in board[row_index]:
        return False

    # Check column
    if cell_value in list(zip(*board))[column_index]:
        return False

    # Check 3x3 box matrix
    affected_row = board[(row_index//3)*3: (row_index//3)*3+3]
    matrix = list(zip(*affected_row))[(column_index//3)*3: (column_index//3)*3+3]
    if any(cell_value in row for row in matrix):
        return False

    return True


board = [
[1, 2, 3, 4, 0, 0, 5, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[6, 0, 0, 0, 7, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 3, 0, 0],
[0, 0, 0, 0, 8, 0, 0, 0, 0],
[8, 0, 0, 0, 0, 0, 0, 0, 7],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 5, 0, 0, 0, 0, 0]
]


print(backtracking(board))
print(board)
print("DONE")