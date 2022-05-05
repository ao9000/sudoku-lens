"""
    Uses backtracking to obtain the fastest solution to a given valid sudoku puzzle
"""


import numpy as np
import cv2
from backtracking import backtracking, create_empty_board, BLANK_STATE
from copy import deepcopy


def draw_puzzle(board):
    """
    Renders the sudoku puzzle using OpenCV

    Renders:
    1. 9 vertical lines
    2. 9 horizontal lines
    3. Cell values

    :param board: type: list
    9x9 nested list simulating the sudoku board

    :return: type: numpy.ndarray
    Image of the rendered sudoku puzzle
    """
    # Draw board
    # First, create a black mat image
    board_image = np.zeros((600, 600), np.uint8)

    # Second, draw sudoku lines
    for num in range(1, 9):
        if num % 3:
            # Horizontal
            cv2.line(board_image, (0, (600//9)*num), (600, (600//9)*num), (255, 255, 255), 2)
            # Vertical
            cv2.line(board_image, ((600 // 9) * num, 0), ((600 // 9) * num, 600), (255, 255, 255), 2)
        else:
            # Every 3rd line draw line with more thickness
            # Horizontal
            cv2.line(board_image, (0, (600//9)*num), (600, (600//9)*num), (255, 255, 255), 5)
            # Vertical
            cv2.line(board_image, ((600 // 9) * num, 0), ((600 // 9) * num, 600), (255, 255, 255), 5)

    # Third, draw number in board
    for row in range(0, 9):
        for col in range(0, 9):
            # Check if cell is not a blank state (0)
            if board[row][col] != BLANK_STATE:
                cv2.putText(board_image, str(board[row][col]), ((600//9)*col+(600//27), (600//9)*row+(600//14)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
    return board_image


def main():
    """
    Shows how backtracking is applied
    Renders the unsolved and solution of the given sudoku puzzle
    """

    # Create a blank Sudoku board
    board = create_empty_board()

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

    # board[0] = [0, 0, 0, 2, 6, 0, 7, 0, 1]
    # board[1] = [6, 8, 0, 0, 7, 0, 0, 9, 0]
    # board[2] = [1, 9, 0, 0, 0, 4, 5, 0, 0]
    # board[3] = [8, 2, 0, 1, 0, 0, 0, 4, 0]
    # board[4] = [0, 0, 4, 6, 0, 2, 9, 0, 0]
    # board[5] = [0, 5, 0, 0, 0, 3, 0, 2, 8]
    # board[6] = [5 , 1 , 9 , 3 , 2 , 6 , 8 , 7 , 4]
    # board[7] = [0, 4, 0, 0, 5, 0, 0, 3, 6]
    # board[8] = [7, 0, 3, 0, 1, 8, 0, 0, 0]

    # Removing Completed 5 and 6 causing issues for most_constraint

    solved_board, steps = backtracking(deepcopy(board))

    
    print_board(board)

    print("Solved in " + str(steps))
    print_board(solved_board)
    if False:
        # Show unsolved puzzle
        cv2.imshow("Unsolved", draw_puzzle(board))
        cv2.waitKey(0)

        if steps:
            # Show solved puzzle
            cv2.imshow(f"Solved in {steps} steps", draw_puzzle(solved_board))
            cv2.waitKey(0)
        else:
            print("Invalid puzzle")

        # Close all windows
        cv2.destroyAllWindows()

def print_board(board):
    for row in range(9):
        for col in range(9):
            print(str(board[row][col]) + " , ", end="")
        print("")

if __name__ == '__main__':
    main()
