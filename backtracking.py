import numpy as np
import math
from copy import deepcopy

BLANK_STATE = 0


def backtracking(board, index):
    row_index, column_index = index

    if is_board_full(board) or row_index == 9:
        print("Finished")
        print(board)
        exit()
        return board

    # Check if cursor is at a blank state
    if board[row_index, column_index] == BLANK_STATE:
        # Blank state
        # Assign number to box
        for num in range(1, 10):
            board[row_index, column_index] = num

            # Check if valid
            if check_valid(board, (row_index, column_index)):
                print("{} Valid".format(num))
                # Valid, move to next box
                board = backtracking(board, get_next_index(index))

            else:
                print("{} Not valid".format(num))
                # Not valid
                if num == 9:
                    # Every possible number not valid
                    # Perform backtrack
                    board[row_index, column_index] = BLANK_STATE
                    print("Backtrack")
                    return board
    else:
        # Not blank state, default values
        # Shifting to next box
        board = backtracking(board, get_next_index(index))

    return board


def check_valid(board, index):
    row_index, column_index = index

    # Check if number is valid
    # Check row
    row = np.delete(board[row_index], obj=column_index)
    if board[row_index, column_index] in row:
        return False

    # Check column
    column = np.delete(board[:, column_index], obj=row_index)
    if board[row_index, column_index] in column:
        return False

    # Check 3x3 box area
    box_row_index = (math.ceil((row_index + 1) / 3) - 1) * 3
    box_column_index = (math.ceil((column_index + 1) / 3) - 1) * 3

    box_row_index_relative = row_index % 3
    box_column_index_relative = column_index % 3

    box = np.delete(np.ravel(board[box_row_index:box_row_index + 3, box_column_index:box_column_index + 3], order='C'),
                    obj=box_row_index_relative * 3 + box_column_index_relative)
    if board[row_index, column_index] in box:
        return False

    return True


def is_board_full(board):
    if BLANK_STATE in board:
        return False

    return True


def get_next_index(index):
    row_index, column_index = index

    if column_index == 8:
        # Next row
        return row_index + 1, 0
    else:
        # Next column
        return row_index, column_index + 1
