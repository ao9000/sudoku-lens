from tests.puzzles import puzzle_kaggle, puzzle_17clue, puzzle_forum_hardest, puzzle_gen_invalid
from backtracking import backtracking
BLANK_STATE = 0


def test_puzzle_gen_invalid(puzzle_gen_invalid):
    for puzzle in puzzle_gen_invalid[:10]:
        solved, step = backtracking(puzzle)

        assert step == 0
        assert solved == puzzle


def test_puzzle_kaggle(puzzle_kaggle):
    for puzzle in puzzle_kaggle[:10]:
        solved, step = backtracking(puzzle)

        assert step != 0 and not any(BLANK_STATE in row for row in solved)


def test_puzzle_17clue(puzzle_17clue):
    for puzzle in puzzle_17clue[:10]:
        solved, step = backtracking(puzzle)

        assert step != 0 and not any(BLANK_STATE in row for row in solved)


def test_puzzle_forum_hardest(puzzle_forum_hardest):
    for puzzle in puzzle_forum_hardest[:10]:
        solved, step = backtracking(puzzle)

        assert step != 0 and not any(BLANK_STATE in row for row in solved)
