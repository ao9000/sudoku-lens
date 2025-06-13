"""
Algorithm X with Dancing Links for Sudoku Solver

References:
https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf
https://youtu.be/_cR9zDlvP88?si=YQDShH4zwuaCqzaW
https://stackoverflow.com/questions/1518335/the-dancing-links-algorithm-an-explanation-that-is-less-explanatory-but-more-o
https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
"""


from itertools import product
import numpy as np
from copy import deepcopy

BLANK_STATE = 0

def algorithm_x_dl(size, grid):
    # Make sure grid is a numpy array
    grid = np.array(grid, dtype=int)

    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])
    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C) # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)

    # Python list version
    # for i, row in enumerate(grid):
    #     for j, n in enumerate(row):
    #         if n:
    #             select(X, Y, (i, j, n))

    # Numpy version
    for r in range(N):
        for c in range(N):
            n = int(grid[r, c])
            if n:
                select(X, Y, (r, c, n))

    for solution in solve(X, Y, []):
        # Python list version
        # for (r, c, n) in solution:
        #     grid[r][c] = n
        # yield grid

        # Numpy version
        for (r, c, n) in solution:
            grid[r, c] = n
        yield grid

def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y

def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


if __name__ == "__main__":
    # Sample of invalid board
    # board = np.array([
    #     [5, 9, 0, 0, 7, 0, 0, 0, 0],
    #     [6, 0, 0, 1, 3, 5, 0, 0, 0],
    #     [0, 7, 8, 0, 0, 0, 0, 6, 0],
    #     [8, 0, 0, 0, 6, 0, 0, 0, 3],
    #     [4, 0, 0, 8, 0, 3, 0, 0, 1],
    #     [7, 0, 0, 0, 2, 0, 0, 0, 6],
    #     [0, 6, 0, 0, 0, 0, 2, 8, 0],
    #     [0, 0, 0, 4, 1, 9, 0, 0, 5],
    #     [0, 0, 0, 0, 8, 0, 0, 7, 9],
    # ])

    # # Sample of invalid board, starting invalid, key error
    # board = np.array([
    #     [5, 9, 5, 0, 7, 0, 0, 0, 0],
    #     [6, 0, 0, 1, 3, 5, 0, 0, 0],
    #     [0, 7, 8, 0, 0, 0, 0, 6, 0],
    #     [8, 0, 0, 0, 6, 0, 0, 0, 3],
    #     [4, 0, 0, 8, 0, 3, 0, 0, 1],
    #     [7, 0, 0, 0, 2, 0, 0, 0, 6],
    #     [0, 6, 0, 0, 0, 0, 2, 8, 0],
    #     [0, 0, 0, 4, 1, 9, 0, 0, 5],
    #     [0, 0, 0, 0, 8, 0, 0, 7, 9],
    # ])

    # Valid board example
    board = np.zeros((9, 9), dtype=int)

    board[0] = [0, 0, 0, 2, 6, 0, 7, 0, 1]
    board[1] = [6, 8, 0, 0, 7, 0, 0, 9, 0]
    board[2] = [1, 9, 0, 0, 0, 4, 5, 0, 0]
    board[3] = [8, 2, 0, 1, 0, 0, 0, 4, 0]
    board[4] = [0, 0, 4, 6, 0, 2, 9, 0, 0]
    board[5] = [0, 5, 0, 0, 0, 3, 0, 2, 8]
    board[6] = [0, 0, 9, 3, 0, 0, 0, 7, 4]
    board[7] = [0, 4, 0, 0, 5, 0, 0, 3, 6]
    board[8] = [7, 0, 3, 0, 1, 8, 0, 0, 0]

    generator = algorithm_x_dl((3, 3), deepcopy(board))
    try:
        solved = next(generator, None)
    except KeyError:
        print("Invalid Sudoku board configuration to begin with.")
        solved = None

    if solved is None:
        print("No solution found.")
    else:
        print("Solution found:")
        print(solved)


