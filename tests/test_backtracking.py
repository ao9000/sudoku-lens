from backtracking import backtracking
from tests.puzzles import easy_puzzles


def test_easy_puzzles(easy_puzzles):
    # Unpack
    puzzles, solutions = easy_puzzles

    for puzzle, solution in zip(puzzles, solutions):
        solved = backtracking(puzzle)

        assert (solved == solution).all()
