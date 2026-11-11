# Image Enhancement Pipeline
This project implements a classical image processing pipeline to enhance degraded images (low contrast, non-uniform illumination and noise) using OpenCV and NumPy. 

## Project Structure
The project is structured as it follows:
- `HW1_ImageEnhancement/` is the main folder that contains all the project and into that you can find
	- `data/` that has the original and the degraded images divided properly in two folders
    - `report/` that contain the MD and the PDF report
	- `results/` contains a side-by-side visual comparisons of the images and the `metrics.csv` file with PSNR and MSE values
	- `src/` is the folder with all the Python files divided in:
		- `degrade.py` that has the job to degrade versions of the original image
		- `methods.py` core image enhancement functions
		- `visualize.py` generates side-by-side comparative figures
		- `evaluate.py` computes PSNR and MSE metrics and saves to CSV
		- `run.py` run the complete pipeline from start to the finish
- `requirements.txt` is the file with all the dependencies for Python

## Requirements
The project uses Python 3.15 installed forcefully and then I activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The required packages are NumPy, OpenCV and Matplotlib

## How to run
1. Clone this repository and navigate to the root
2. Create a virtual environment as said before on [[#Requirements]]
3. Change directory into `HW1_ImageEnhancement`
   ```bash
   cd HW1_ImageEnhancement
   ```
4. Run the pipeline with a single command
   ```bash
   python src/run.py
   ```
5. The results are generated in:
	1. `results/images/` for the comparison side-by-side
	2. `results/metrics` for the quantitative evaluation

## Degradation Parameters
The synthetic degradation is applied with fixed parameters for reproducibility
- `ALPHA = 0.5` and `BETA = 30` for the contrast reduction
- `MEAN = 0`, `STD = 10` and `SEED = 42` for the Gaussian noise