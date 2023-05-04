Some instructions for running the notebook:

1. Create a virtual environment using the following command in PowerShell
    py -m venv <directory>
2. Initialise the virtual environment using the following command in PowerShell
    <directory>\Scripts\Activate.ps1
3. Run the following command with your virtual environment active to install all required dependencies
    py -m pip install -r requirements.txt
4. Check that the notebook's Python kernel is switched to the virtual environment Python kernel!


For dev work:

- Remember to run the following command in your virtual environment (it'll save installed packages used in dev work)
    py -m pip freeze > requirement.txt