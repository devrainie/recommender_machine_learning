# 1. A local program:
- The files needed for this project are the styles csv files and its images. These files altogether are very large to upload here so they can be downloaded from kaggle: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- The name of the styles csv variable may need to be changed to the one downloaded. The styles file and images will need to go into the "dataset_files" folder.

# 2. Download python if not already downloaded 
- via https://www.python.org/downloads/

# 3. Download anaconda if not already downloaded
- via https://www.anaconda.com/download/

# 4. Open an IDE 
- such as visual studio code and open the ‘final year project’ folder.

# 5. Change directories 
- 5.	You should see ‘(base)’ in the terminal which means anaconda is installed. Change directories into the final year project folder using 'cd' in the IDE inside the IDE terminal.

# 6. Activate anaconda base
- Type ‘conda activate base’ in the IDE terminal

# 7. Create new conda environment
- Then type ‘conda create -n myenv’. Type ‘y’ and press enter to proceed

# 8. Activate new env
- Then type ‘conda activate myenv’

# 9. Check the interpreter of the IDE matches the new anaconda environment created
- In the IDE, the interpreter should include anaconda and the chosen environment created, ‘myenv’ otherwise an error will be thrown. If the correct interpreter is not seen, it should look something like ‘../anaconda3/envs/myenv’. Refresh if the path is not available and it should appear. Select this interpreter.

# 10. Run all installs so imports will be recognised
-	In the IDE terminal with the ‘myenv’ environment activated, run these commands one after the other:
- ‘pip install pillow’
- ‘pip install streamlit’
- ‘pip install sklearn’
- ‘pip install -U scikit-learn’
- ‘pip install numpy’
- ‘pip install num2words’
- ‘pip install pandas’
- ‘pip install streamlit-option-menu’


- Anaconda may already have a few of these libraries but they can be installed if they are not present, or the imports are not working. Once all installed, all import errors should disappear. If not close and reopen the IDE and activate ‘myenv’ again in terminal.

# 11. Change the directory variables to match your own directory
- 11.	IMPORTANT: The directory variables called ‘dir’ and ‘dir_img’ inside the final_recommender.py and addItem.py should be changed to match the directory of your machine. Copy the path (of the downloaded files) for the ‘dataset_files’ folder to save in the ‘dir’ variable. And copy the file path for the images folder to save in ‘dir_img’. This is where the styles_2.csv and image file are located.

- Line 36 and 38 contain the variables in the final_recommender.py file. And line 12 contains the variable in the addItem.py file.

# 12. Run streamlit in terminal
- Then run ‘streamlit run home.py’ This should be run in the dicrectory containing the ‘home.py’ file which should be final_year_project. This should fire up a new tab with the Streamlit user interface on a local host. 

The structure of the files should be like below and should include the main files home.py, addItem.py and final_recommender.py. Along with all the images inside the images folder:

Final_year_project
|-- home.py
|--Readme.md
|-- dataset_files
	|--images
	|--styles_2.csv
|--pages
	|--final_recommender.py
	|--addItem.py
