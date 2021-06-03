# Sudoku solver
This project is the implementation of the sudoku solver using OpenCV, Tensorflow, Keras, Backtracking algorithm

## Screenshots
### run_backtracking_demo.py
Backtracking before and after
![run_backtracking_demo.py](media/backtracking-demo-unsolved.png?raw=true "Unsolved board")
![run_backtracking_demo.py](media/backtracking-demo-solved.png?raw=true "Solved board")

### run_grid_extraction.py
Using OpenCV to detect sudoku grid and cells
![run_grid_extraction.py](media/grid-extractor-demo-original.png?raw=true "Original image")
![run_grid_extraction.py](media/grid-extractor-demo-extracted.png?raw=true "Extracted grid cells")

### run_cell_extraction.py
After cell detection, extract cells to prepare them for testing data
![run_cell_extraction.py](media/cell-extraction-results.png?raw=true "Extracted cells")


### digit-classifier/train.py
Using handwritten mnist dataset to train a simple deep learning model to classify digits
![digit-classifier/train.py](media/digit-classifier-train.png?raw=true "Training a digit classifier model")

### digit-classifier/test.py
Evaluating the trained model using the test dataset
![digit-classifier/test.py](media/digit-classifier-test.png?raw=true "Evaluate the classifier")

### run_sudoku_solver.py
Using all the above techniques, solve a raw sudoku image
![run_sudoku_solver.py](media/unsolved_puzzle.jpg?raw=true "Sample unsolved sudoku image")
![run_sudoku_solver.py](media/solved_puzzle.png?raw=true "Solved sudoku image")

## Core framework/concepts used
- Backtracking

Backtracking algorithm is a type of brute force search.

Process:
1. Searches for a blank cell row first, followed by column
1. Once found a blank cell, begin testing for digits from 1-9 validity on the cells
1. Once a valid number is found, assign the number to the cell and move to the next available cell (Repeat step 1)
    1. If no valid number is found, perform backtrack. Step back to the previous available cell to continue testing for more possible numbers
1. Do this until it reaches the last cell. Then the puzzle will be solved. If no solution is found, return the same board untouched

- Pytest
  
Used to test if the backtracking algorithm is working as intended without any bugs

- OpenCV

Used to extract the sudoku grid, cells and other image processing operations that smoothen the extraction process

- Tensorflow & Keras

Used to train and evaluate a digit classifier

- Matplotlib

Used to draw the training accuracy and loss graphs

- Sklearn

Used to output model accuracy scores and confusion matrix

## Getting started
Follow the steps in order below

### Prerequisites
You will need to have these installed before doing anything else

- Python- 3.9.5 and above https://www.python.org/downloads/

### Installation
- Installing Python packages
```
# cd into the root folder of the project
# Change the path accordingly to your system
cd /home/sudoku-solver

# You should have pip installed as it comes with installing Python
# Installing python packages
pip install -r requirements.txt
```

### Assets
You can download the assets that i have used at the following link
https://drive.google.com/file/d/16VNkbgs-DNK7YlmFRN4my7bunQAbouqc/view?usp=sharing

The assets file contain:
1. Model files
2. Model test dataset that I have curated
3. Sample unsolved and solved sudoku images 

## Usage
- Run backtracking demo
```
# Make sure your in the root directory of the project
python run_backtracking_demo.py

# Results will pop up
```

- Run cell extraction
```
# Make sure your in the root directory of the project
# Create a new directory for the output files
mkdir cells
python run_cell_extraction.py

# You should be able to see the extracted cells
```

-  Run grid extractor demo
```
# Make sure your in the root directory of the project
python run_grid_extractor_demo.py

# Results will pop up
```

- Run Sudoku solver
```
# Make sure your in the root directory of the project
mkdir images
mkdir images/unsolved
mkdir images/solved

# Be sure to store the unsolved sudoku images in images/unsolved directory before running
python run_sudoku_solver.py

# Results will be stored in images/solved directory
```

- Digit classifier, Run training
```
# Make sure your in the root directory of the project
cd digit-classifier
mkdir models

python train.py

# After training is done, model file will be saved into digit-classifier/models
```

- Digit classifier, Run training
```
# Make sure your in the root directory of the project
cd digit-classifier
mkdir test

# Be sure to store all the testing digit images into their respective directories digit-classifier/test/{digit}

python test.py

# Results will be shown after script completion
```

## References
References
Brownlee, J. (2019). How to Develop a CNN for MNIST Handwritten Digit Classification. Machine Learning Mastery. Retrieved 3 June 2021, from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/.

Find sudoku grid using OpenCV and Python. Stack Overflow. (2018). Retrieved 3 June 2021, from https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python.

Malo, n., & Maisonneuve. (2020). How to get the cells of a sudoku grid with OpenCV?. Stack Overflow. Retrieved 3 June 2021, from https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv.

Sinha, U. (2017). SuDoKu Grabber in OpenCV: Grid detection - AI Shack. Aishack.in. Retrieved 3 June 2021, from https://aishack.in/tutorials/sudoku-grabber-opencv-detection/.