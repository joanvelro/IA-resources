# IA_resources

    * Plotly functions to plot
    * Communications settigns
    * Pandas functions
    * Time series functions

* Use this repository as a submodule within the repository of the project in which it will be applied

```
git submodule add https://github.com/joanvelro/IA-resources.git
```

* it is added the file .gitmodules 

```
[submodule IA-resources]
path = IA-resources
url = https://github.com/joanvelro/IA-resources.git
```

# Set up IA-ready environemnt

## Install prerequisites
First steps (instructions for Mac or Linux). You need to install a recent version of Python, plus the packages 'keras', 'numpy', 'matplotlib' and 'jupyter'.

### Install a recent Python

If you haven't installed a recent Python I recommend installing via Homebrew on a Mac from http://brew.sh and then installing Python via 

```
brew install python
```

### Configure a virtual environment

You can install the packages globally, but I suggest installing them in a 'virtualenv' virtual environment that basically encapsulates a full isolated Python environment.
First you'll need to install a Python package manager called 'pip' thus:

```
easy_install pip
``` 
(If you get a permissions error, try adding a 'sudo' to the beginning, so 'sudo easy_install pip')
```
sudo easy_install pip
``` 
Now install virtualenv thus:
```
pip install virtualenv
```
Navigate to your home directory 'cd ~' and create a virtual environment. We'll call it 'my_virtual_env'
```
virtualenv my_virtual_env
```
    
Now, to switch your shell environment to be within the env:
```
source my_virtual_env/bin/activate
```  
 
```
my_virtual_env/Scripts/activate
```
Great: now you can install the other prerequisites into this environment.
```
pip install numpy jupyter keras matplotlib tensorflow xgboost pyomo statsmodels
```

Export environmnet

```
pip freeze > requirements.txt
```
This file can then be used by collaborators to update virtual environments using the following command.
```
pip install -r requirements.txt
```

## Open a new notebook

Now that everything's installed, you can open one of these web-based Python environments with the following command:
```
jupyter notebook
```
Create a new Python notebook from the "New" menu at the top-right:

You should now be able to run Python in your browser!

Deactivate the Environment

To return to normal system settings, use the deactivate command.
```
deactivate
```

(C) Jose Angel Velasco (@joanvelro) - 2021
