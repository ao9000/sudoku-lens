"""
    Test cases to validify if the backtracking algorithm is working as intended

    Test includes:
    1. Invalid cases (Algorithm should return 0 steps
    2. Easy puzzles
    3. Intermediate puzzles
    4. Difficult puzzles
    5. Not fun puzzles
"""


from backtracking import backtracking
from tests.puzzles import easy_puzzles, intermediate_puzzles, difficult_puzzles, not_fun_puzzles, invalid_puzzles


def test_invalid_puzzles(invalid_puzzles):
    for puzzle in invalid_puzzles:
        solved, step = backtracking(puzzle)

        assert (solved == puzzle).all()
        assert step == 0


def test_easy_puzzles(easy_puzzles):
    # Unpack
    puzzles, solutions = easy_puzzles

    for puzzle, solution in zip(puzzles, solutions):
        solved, step = backtracking(puzzle)

        assert (solved == solution).all()
        assert step != 0


def test_intermediate_puzzles(intermediate_puzzles):
    # Unpack
    puzzles, solutions = intermediate_puzzles

    for puzzle, solution in zip(puzzles, solutions):
        solved, step = backtracking(puzzle)

        assert (solved == solution).all()
        assert step != 0


def test_difficult_puzzles(difficult_puzzles):
    # Unpack
    puzzles, solutions = difficult_puzzles

    for puzzle, solution in zip(puzzles, solutions):
        solved, step = backtracking(puzzle)

        assert (solved == solution).all()
        assert step != 0


def test_not_fun_puzzles(not_fun_puzzles):
    # Unpack
    puzzles, solutions = not_fun_puzzles

    for puzzle, solution in zip(puzzles, solutions):
        solved, step = backtracking(puzzle)

        assert (solved == solution).all()
        assert step != 0
