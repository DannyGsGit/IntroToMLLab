# Intro To ML Lab
Welcome to the intro to ML lab! In this lab, we will step through a simple toy example to introduce basic concepts and common pitfalls of machine learning.

# Getting Started
For the workshop, you will be provided an account on our JupyterHub server. Simply visit the URL provided by the administrator and enter your credentials when prompted.

## Clone This Repo
To get the code contained in this repo, follow these steps:
* From the JupyterHub landing page, click "New" > "Terminal". This will open a command line.
* In the command line, type or paste the snippet below. Tip: Use Shift-Insert to paste in this command line, Ctrl-V will not work!
```
git clone https://github.com/DannyGsGit/IntroToMLLab.git
```
* You will see the code download to your environment. To confirm that the directory exists, type 'ls' into the command line.
* Navigate back to the Jupyter landing page (the tab should remain open), where you will see the folder.
* Navigate to the following path and open the Home Prices notebook: ./IntroToMLLab/Code/HomePrices.ipynb

# Appendix

## Getting Kaggle Leaderboard
Install the [Kaggle API](https://github.com/Kaggle/kaggle-api), then run the following command to download the leaderboard:
```
kaggle competitions leaderboard house-prices-advanced-regression-techniques -d
```
