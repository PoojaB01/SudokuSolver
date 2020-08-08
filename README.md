# Sudoku Solver Camera

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

The application lets you click picture of a sudoku puzzle and gives it's solution.

## Working

Given Image
<br />
<img src = "Sudoku Solver/images/sudoku4.jpg" width = 300px style = "padding:30px;"></img>

1. Detection of Sudoku Bounding Box and extraction of puzzle image from original image. <br />
<img src = "Sudoku Solver/images/original.jpg" width = 300px style = "padding:30px;"></img>
2. Image processing and detection of digits. <br />
<img src = "Sudoku Solver/images/digit106.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit100.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit107.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit102.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit101.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit118.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit110.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit103.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "Sudoku Solver/images/digit105.jpg" width = 30px style = "padding:2px; display: inline"></img>
3. Classifying digits based on their images using a KNN - model.
4. Solving the puzzle using backtracking. <br />
<img src = "Sudoku Solver/images/solved.jpg" width = 300px style = "padding:30px;"></img>
5. Printing solution back on image. <br />
<img src = "Sudoku Solver/images/final.jpg" width = 300px style = "padding:30px;"></img>

Made by : Pooja Bhaagt 
