import numpy as np
from backtracking import backtracking


def create_board():
    # Create a blank Sudoku board
    board = np.zeros((9, 9), dtype=int)

    # Sample Board config
    board[0] = [0, 0, 0, 2, 6, 0, 7, 0, 1]
    board[1] = [6, 8, 0, 0, 7, 0, 0, 9, 0]
    board[2] = [1, 9, 0, 0, 0, 4, 5, 0, 0]
    board[3] = [8, 2, 0, 1, 0, 0, 0, 4, 0]
    board[4] = [0, 0, 4, 6, 0, 2, 9, 0, 0]
    board[5] = [0, 5, 0, 0, 0, 3, 0, 2, 8]
    board[6] = [0, 0, 9, 3, 0, 0, 0, 7, 4]
    board[7] = [0, 4, 0, 0, 5, 0, 0, 3, 6]
    board[8] = [7, 0, 3, 0, 1, 8, 0, 0, 0]

    return board


def main():
    print(backtracking(create_board()))


if __name__ == '__main__':
    main()
