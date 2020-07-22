"""
    Puzzles & their solutions
    Puzzles retrieved from https://dingo.sbs.arizona.edu/~sandiway/sudoku/examples.html
"""

import pytest
import numpy as np

BLANK_STATE = 0


# Easy puzzles
@pytest.fixture
def easy_puzzles():
    easy1 = np.array([[BLANK_STATE, BLANK_STATE, BLANK_STATE, 2, 6, BLANK_STATE, 7, BLANK_STATE, 1],
                      [6, 8, BLANK_STATE, BLANK_STATE, 7, BLANK_STATE, BLANK_STATE, 9, BLANK_STATE],
                      [1, 9, BLANK_STATE, BLANK_STATE, BLANK_STATE, 4, 5, BLANK_STATE, BLANK_STATE],
                      [8, 2, BLANK_STATE, 1, BLANK_STATE, BLANK_STATE, BLANK_STATE, 4, BLANK_STATE],
                      [BLANK_STATE, BLANK_STATE, 4, 6, BLANK_STATE, 2, 9, BLANK_STATE, BLANK_STATE],
                      [BLANK_STATE, 5, BLANK_STATE, BLANK_STATE, BLANK_STATE, 3, BLANK_STATE, 2, 8],
                      [BLANK_STATE, BLANK_STATE, 9, 3, BLANK_STATE, BLANK_STATE, BLANK_STATE, 7, 4],
                      [BLANK_STATE, 4, BLANK_STATE, BLANK_STATE, 5, BLANK_STATE, BLANK_STATE, 3, 6],
                      [7, BLANK_STATE, 3, BLANK_STATE, 1, 8, BLANK_STATE, BLANK_STATE, BLANK_STATE]])

    easy1_solution = np.array([[4, 3, 5, 2, 6, 9, 7, 8, 1],
                               [6, 8, 2, 5, 7, 1, 4, 9, 3],
                               [1, 9, 7, 8, 3, 4, 5, 6, 2],
                               [8, 2, 6, 1, 9, 5, 3, 4, 7],
                               [3, 7, 4, 6, 8, 2, 9, 1, 5],
                               [9, 5, 1, 7, 4, 3, 6, 2, 8],
                               [5, 1, 9, 3, 2, 6, 8, 7, 4],
                               [2, 4, 8, 9, 5, 7, 1, 3, 6],
                               [7, 6, 3, 4, 1, 8, 2, 5, 9]])

    easy2 = np.array([[1, BLANK_STATE, BLANK_STATE, 4, 8, 9, BLANK_STATE, BLANK_STATE, 6],
                      [7, 3, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 4, BLANK_STATE],
                      [BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 1, 2, 9, 5],
                      [BLANK_STATE, BLANK_STATE, 7, 1, 2, BLANK_STATE, 6, BLANK_STATE, BLANK_STATE],
                      [5, BLANK_STATE, BLANK_STATE, 7, BLANK_STATE, 3, BLANK_STATE, BLANK_STATE, 8],
                      [BLANK_STATE, BLANK_STATE, 6, BLANK_STATE, 9, 5, 7, BLANK_STATE, BLANK_STATE],
                      [9, 1, 4, 6, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE],
                      [BLANK_STATE, 2, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 3, 7],
                      [8, BLANK_STATE, BLANK_STATE, 5, 1, 2, BLANK_STATE, BLANK_STATE, 4]])

    easy2_solution = np.array([[1, 5, 2, 4, 8, 9, 3, 7, 6],
                               [7, 3, 9, 2, 5, 6, 8, 4, 1],
                               [4, 6, 8, 3, 7, 1, 2, 9, 5],
                               [3, 8, 7, 1, 2, 4, 6, 5, 9],
                               [5, 9, 1, 7, 6, 3, 4, 2, 8],
                               [2, 4, 6, 8, 9, 5, 7, 1, 3],
                               [9, 1, 4, 6, 3, 7, 5, 8, 2],
                               [6, 2, 5, 9, 4, 8, 1, 3, 7],
                               [8, 7, 3, 5, 1, 2, 9, 6, 4]])

    easy_puzzles = [easy1, easy2]
    easy_puzzles_solutions = np.array([easy1_solution, easy2_solution])

    return easy_puzzles, easy_puzzles_solutions


@pytest.fixture
def intermediate_puzzles():
    intermediate1 = np.array([[BLANK_STATE, 2, BLANK_STATE, 6, BLANK_STATE, 8, BLANK_STATE, BLANK_STATE, BLANK_STATE],
                              [5, 8, BLANK_STATE, BLANK_STATE, BLANK_STATE, 9, 7, BLANK_STATE, BLANK_STATE],
                              [BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 4, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE],
                              [3, 7, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 5, BLANK_STATE, BLANK_STATE],
                              [6, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 4],
                              [BLANK_STATE, BLANK_STATE, 8, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 1, 3],
                              [BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE, 2, BLANK_STATE, BLANK_STATE, BLANK_STATE, BLANK_STATE],
                              [BLANK_STATE, BLANK_STATE, 9, 8, BLANK_STATE, BLANK_STATE, BLANK_STATE, 3, 6],
                              [BLANK_STATE, BLANK_STATE, BLANK_STATE, 3, BLANK_STATE, 6, BLANK_STATE, 9, BLANK_STATE]
                              ])

    intermediate1_solution = np.array([[1, 2, 3, 6, 7, 8, 9, 4, 5],
                                       [5, 8, 4, 2, 3, 9, 7, 6, 1],
                                       [9, 6, 7, 1, 4, 5, 3, 2, 8],
                                       [3, 7, 2, 4, 6, 1, 5, 8, 9],
                                       [6, 9, 1, 5, 8, 3, 2, 7, 4],
                                       [4, 5, 8, 7, 9, 2, 6, 1, 3],
                                       [8, 3, 6, 9, 2, 4, 1, 5, 7],
                                       [2, 1, 9, 8, 5, 7, 4, 3, 6],
                                       [7, 4, 5, 3, 1, 6, 8, 9, 2]
                                       ])

    intermediate_puzzles = [intermediate1]
    intermediate_puzzles_solutions = [intermediate1_solution]

    return intermediate_puzzles, intermediate_puzzles_solutions
