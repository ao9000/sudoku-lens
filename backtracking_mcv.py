"""
    Contains backtracking algorithm used for Sudoku

    Backtracking algorithm is a type of brute force search.
    Process:
    1. Searches for a blank cell row first, followed by column
    2. Once found a blank cell, begin testing for digits from 1-9 validity on the cells
    3a. Once a valid number is found, assign the number to the cell and move to the next available cell (Repeat step 1)
    3b. If no valid number is found, perform backtrack. Step back to the previous available cell to continue testing for
    more possible numbers
    4. Do this until it reaches the last cell. Then the puzzle will be solved. If no solution is found, return the same board untouched
"""


BLANK_STATE = 0
constraintRow = [[] for i in range(9)]
constraintCol = [[] for i in range(9)]
constraintBox = [[] for i in range(9)]
constraintCal = False

def create_empty_board():
    """
    Create a 9x9 nested list of BLANK_STATE to simulate the sudoku board

    :return: type: list
    9x9 Nested list containing BLANK_STATE
    """
    return [[BLANK_STATE] * 9 for _ in range(0, 9)]


def backtracking_constraint(board, step=1):
    """
    Recursive function for the backtracking algorithm

    Process:
    1. Searches for a blank cell row first, followed by column
    2. Once found a blank cell, begin testing for digits from 1-9 validity on the cells
    3a. Once a valid number is found, assign the number to the cell and move to the next available cell (Repeat step 1)
    3b. If no valid number is found, perform backtrack. Step back to the previous available cell to continue testing for
    more possible numbers
    4. Do this until it reaches the last cell. Then the puzzle will be solved. If no solution is found, return the same board untouched

    :param board: type: list
    The 9x9 nested list simulating sudoku board

    :param step: type: int
    The depth of the current recursion

    :return: type: tuple
    Tuple containing the board and step
    """

    if board_cleared():
        return board, step-1

    # index = get_next_index(board)
    index = get_most_constraint(board)
    # return board, 0 # Temp pause

    # Check if index was found
    if index:
        # Unpack index
        row_index, column_index = index
        # Get Valid Values
        num_arr = return_valid_values(row_index, column_index)
        for num in num_arr:
            # Valid, assign number to cell
            board[row_index][column_index] = num
            add_constraint(row_index, column_index, num)

            # Move on to next available cell
            board, step = backtracking_constraint(board, step+1)

            if (board_cleared()):
                return board, step

            remove_constraint(row_index, column_index, board[row_index][column_index])

        board[row_index][column_index] = BLANK_STATE
        return board, step-1
    else:
        # No empty cell, meaning puzzle has been completed
        return board, step-1

def board_cleared():
    global constraintBox
    for i in range(9):
        if (len(constraintBox[i]) < 9):
            return False
    
    return True

def return_valid_values(row, col):
    global constraintBox, constraintCol, constraintRow
    box = (int(row/3)*3) + int(col/3)
    rtn_arr = [*range(1,10)]
    constraintList = set(constraintBox[box] + constraintRow[row] + constraintCol[col])
    for i in constraintList:
        rtn_arr.remove(i)
    return rtn_arr

def calculate_constraints(board):
    global constraintCal, constraintBox, constraintCal, constraintCol
    for row_index in range(0, 9):
            for column_index in range(0, 9):
                if board[row_index][column_index] != BLANK_STATE:
                    val = board[row_index][column_index]
                    constraintRow[row_index].append(val)
                    constraintCol[column_index].append(val)
                    box = (int(row_index/3)*3) + int(column_index/3)
                    constraintBox[box].append(val)

    constraintCal = True

def get_most_constraint(board):
    # Most Constraint box for now :(
    global constraintCal, constraintBox, constraintCal, constraintCol
    if not constraintCal:
        calculate_constraints(board)

    mostConstraintBox = 0
    for box_index in range(1,9):
        if len(constraintBox[mostConstraintBox]) < len(constraintBox[box_index]) or len(constraintBox[mostConstraintBox]) >= 9:
            if len(constraintBox[box_index]) < 9:
                mostConstraintBox = box_index
    startRow = int(mostConstraintBox / 3)*3
    startCol = (mostConstraintBox%3)*3

    for row_index in range(startRow, startRow+3):
        for column_index in range(startCol, startCol+3):
            if board[row_index][column_index] == BLANK_STATE:
                return row_index, column_index
    
    print("Cant find any")
    return 0,0

def add_constraint(row, col, cell_value):
    global constraintBox, constraintRow, constraintCol
    # Add new value to constraint tracking
    constraintRow[row].append(cell_value)
    constraintCol[col].append(cell_value)

    box = (int(row/3)*3) + int(col/3)
    constraintBox[box].append(cell_value)

def remove_constraint(row, col, cell_value):
    global constraintBox, constraintRow, constraintCol
    # Remove value from constraint tracking
    try:
        constraintRow[row].remove(cell_value)
        constraintCol[col].remove(cell_value)

        box = (int(row/3)*3) + int(col/3)
        constraintBox[box].remove(cell_value)
    except:
        print("ERROR:")
        print(constraintRow[row])
        print(cell_value)