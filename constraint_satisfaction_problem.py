import itertools

BLANK_STATE = 0


def create_empty_board():
    return [[BLANK_STATE] * 9 for _ in range(0, 9)]


def generate_domain_board(board):
    # Start off with all values as possible, slowly delete from list
    possible = [[list(range(1, 10)) for _ in range(0, 9)] for _ in range(0, 9)]

    # Replace already existing variables with None in possibility list
    for row_index, row in enumerate(board, start=0):
        for column_index, col in enumerate(row, start=0):
            if col != BLANK_STATE:
                # Cell is not a BLANK_STATE
                # Part 1
                # Assign NONE to possible list with same indexes
                possible[row_index][column_index] = None

                # Part 2
                # Then delete the appropriate value from the list in respect to row/col/3x3 region
                # Handle rows first
                for value_list in possible[row_index]:
                    try:
                        value_list.remove(col)
                    except ValueError:
                        # Value not in list, ignore
                        pass
                    except AttributeError:
                        # None, not applicable, ignore too
                        pass

                # Handle columns
                for value_list in list(zip(*possible))[column_index]:
                    try:
                        value_list.remove(col)
                    except ValueError:
                        # Value not in list, ignore
                        pass
                    except AttributeError:
                        # None, not applicable, ignore too
                        pass

                # Handle 3x3 box region
                affected_row = possible[(row_index // 3) * 3: (row_index // 3) * 3 + 3]
                matrix = list(zip(*affected_row))[(column_index // 3) * 3: (column_index // 3) * 3 + 3]
                # Flatten all nested list into one list and loop through
                for value_list in list(itertools.chain.from_iterable(matrix)):
                    try:
                        value_list.remove(col)
                    except ValueError:
                        # Value not in list, ignore
                        pass
                    except AttributeError:
                        # None, not applicable, ignore too
                        pass

    return possible


# Most Constrained Variable (MCV) also called Minimum Remaining Values (MRV)
def mrv_heuristic(possible):
    # Get most suitable variable (Box location) and its domain (Possible values)
    # Init temp var to hold the smallest domain size
    # Max number of domain size is 9, therefore put 10
    smallest_domain = 10
    # Init temp var to hold the respective index
    index = None
    # Init temp var to hold domain, to avoid re-calculating later
    domain = None

    # Loop through finding the smallest domain size
    for row_index, row in enumerate(possible, start=0):
        for column_index, col in enumerate(row, start=0):
            if col and len(col) < smallest_domain:
                smallest_domain = len(col)
                domain = col
                index = (row_index, column_index)

                if len(col) == 1:
                    return index, domain

    # After finding, return smallest
    return index, domain


def update_possible_matrix(possible, row_index, column_index, value):
    # Cell is not a BLANK_STATE
    # Part 1
    # Assign NONE to possible list with same indexes
    possible[row_index][column_index] = None

    # Part 2
    # Then delete the appropriate value from the list in respect to row/col/3x3 region
    # Handle rows first
    for value_list in possible[row_index]:
        try:
            value_list.remove(value)
        except ValueError:
            # Value not in list, ignore
            pass
        except AttributeError:
            # None, not applicable, ignore too
            pass

    # Handle columns
    for value_list in list(zip(*possible))[column_index]:
        try:
            value_list.remove(value)
        except ValueError:
            # Value not in list, ignore
            pass
        except AttributeError:
            # None, not applicable, ignore too
            pass

    # Handle 3x3 box region
    affected_row = possible[(row_index // 3) * 3: (row_index // 3) * 3 + 3]
    matrix = list(zip(*affected_row))[(column_index // 3) * 3: (column_index // 3) * 3 + 3]
    # Flatten all nested list into one list and loop through
    for value_list in list(itertools.chain.from_iterable(matrix)):
        try:
            value_list.remove(value)
        except ValueError:
            # Value not in list, ignore
            pass
        except AttributeError:
            # None, not applicable, ignore too
            pass

    return possible

# Most Constrained Variable (MCV) also called Minimum Remaining Values (MRV)

from copy import deepcopy
def csp(board, possible=None, step=1):
    if not possible:
        # First time, generate possible values matrix
        possible = generate_domain_board(board)

    # Get the best variable and its domain
    index, domain = mrv_heuristic(possible)

    if index:
        # Unpack index
        row_index, column_index = index

        # Try all domain values
        for num in domain:
            # No need check if valid since domain values are always valid
            # Assign
            board[row_index][column_index] = num

            # Update possible matrix
            new_possible = update_possible_matrix(deepcopy(possible), row_index, column_index, num)

            # Move on to next best variable
            board, step = csp(board, new_possible, step + 1)

            if not any(BLANK_STATE in row for row in board):
                return board, step

        # Perform backtracking
        board[row_index][column_index] = 0
        return board, step - 1
    else:
        # No empty cell, meaning puzzle has been completed
        return board, step - 1


