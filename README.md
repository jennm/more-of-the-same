# more-of-the-same
Code and data for More of the Same: Persistent Representational Harms Under Increased Representation


### Setup

#### 1. Create a virtual environment

In your project directory, create a virtual environment using the following command:

```
python3 -m venv more-of-the-same
```
This creates a folder named `more-of-the-same` that contains the isolated Python environment.

#### 2. Activate the virtual environment
**macOS / Linux**
```
source more-of-the-same/bin/activate
```
**Windows (PowerShell)**
```
more-of-the-same\Scripts\Activate.ps1
```
**Windows (Command Prompt)**
```
more-of-the-same\Scripts\activate.bat
```
When activated, your terminal prompt should display `(more-of-the-same)` before the command line.

#### 3. Install dependencies
Once the virtual environment is active, install the required packages with:
```
pip install -r requirements.txt
```
This command installs every package and version listed in the `requirements.txt` file.

##### 3a. (Optional) Confirm dependencies
Confirm that all dependencies were installed successfully:
```
pip list
```

#### 4. Deactivate the Virtual Environment
When you’re done working, deactivate the virtual environment with:
```
deactivate
```


### File organization
more-of-the-same/

│

├── data/ # Data used to reproduce the experimental results and generated from the scripts

│ ├── gpt-3.5/ # Persona and biography generations from GPT-3.5

│ ├── gpt-3.5/ # Persona and biography generations from GPT-4o-mini

│ ├── llama-3.1-70b/ # Persona and biography generations from Llama-3.1-70b

│ ├── names_with_dem.csv # Names from the generations

│ ├── occupations_stats_from_winogender.tsv # Statistics from Bureau of Labor and Statistics on female representation in occupations analyzed

│

├── persona-generation-scripts/ # Scripts for generating personas and biographies

│ ├── generate_occupation_personas_gpt.py # Code to generate personas and biographies for GPT-3.5 and GPT-4o-mini

│ ├── generate_occupation_personas_llama3.py # Code to generate personas and biographies for Llama-3.1-70b

│

├── alpha_values_table.md/ # Markdown table of alpha values (code for producing this is in find_optimal_alpha.ipynb)

├── calculate-gender-association-method-statistics.ipynb # Notebook to reproduce gender association statistics 

├── calibrated_marked_words_from_generated_biographies_and_personas.csv # csv file containing calibrated marked words from generated biographies and personas

├── calibrated_marked_words_table.pdf # Table of calibrated marked words from generated biographies and personas

├── calibrated_marked_words.py # Code for the Calibrated Marked words method

├── create_complete_graphs_and_heatmaps.ipynb # Notebook for creating the majority of graphs, heatmaps, and tables in the paper

├── difference_between_calibrated_and_og_marked_words_occupation_specific_investigation.ipynb # Notebook to reproduce difference between calibrated marked words and original marked words introudced by Cheng et al.

├── find_optimal_alpha.ipynb # Notebook to find the optimal value of alpha for the hybrid parameter

├── find_optimal_clusters.py # Code to find optimal \# of clusters for k-means using the Silhouette Score

├── gender-association-method.py # Code for the Gender Association Method

├── metrics.py # Code for the Subset Representation Bias Score 

├── og_marked_words.py # Code for the Marked Words method introduced by Cheng et al. (2023)

├── README.md # This file

├── requirements.txt # List of Python dependencies

├── ss_scores.png # Silhouette Scores generated from `find_optimal_clusters.py`

└── utils.py # Commonly used functions throughout the repo