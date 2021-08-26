# Sudoku Solver

This tool lets you get the solution of a sudoku puzzle from its picture.

## Execution

For building the digit identification model and store them as pkl file, run:  
```
python3 build_model.py --knc_filename=<filename> --rcf_filename=<filename>
```
Once the model is built, generate the solution for a puzzle from its image using:  
```
python3 get_sudoku_solution.py --model_filename=<pkl filepath> --soduko_image=<image path>
```

## Working

Given Image <br />
<img src = "./images/sudoku4.jpg" width = 300px style = "padding:10px;"></img>

1. Detection of Sudoku Bounding Box and extraction of puzzle image from original image.  
<img src = "./images/original.jpg" width = 300px style = "padding:10px;"></img>
2. Image processing and detection of digits.  
<img src = "./images/digit106.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit100.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit107.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit102.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit101.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit118.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit110.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit103.jpg" width = 30px style = "padding:2px; display: inline"></img> <img src = "./images/digit105.jpg" width = 30px style = "padding:2px; display: inline"></img>
3. Classifying digits based on their images using a KNN - model.
4. Solving the puzzle using backtracking.  
<img src = "./images/solved.jpg" width = 300px style = "padding:10px;"></img>
5. Printing solution back on image.  
<img src = "./images/final.jpg" width = 300px style = "padding:10px;"></img>

Made by : Pooja Bhaagt 
