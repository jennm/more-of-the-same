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
