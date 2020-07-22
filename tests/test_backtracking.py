from backtracking import backtracking
from tests.puzzles import easy_puzzles, intermediate_puzzles, difficult_puzzles, not_fun_puzzles


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
