# legislator-arithmetic A neural network approach to ideal point estimation

## Installation instructions

Install the legislator-arithmetic package from github (for example: `git clone
https://github.com/dargyle/legislator-arithmetic.git`)

The `environment.yml` file in the main directory contains a configuration file that will generate a
conda enviroment called `leg_math_new` containing all the packages necessary to run the models. It
requires a pre-existing installation of of the conda package manager, either through
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or full
[Anaconda](https://docs.anaconda.com/anaconda/install/).

Assuming a working conda instance navigate to the package folder and install the `leg_math_new`
environment using this command:

    conda env create -f environment.yml

See the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for additional help with conda.

## Data acquisition and generation

There are several datasets used in this project, all of which can be generated programmatically
(assuming none of the links have broken in the interim...)

- Synthetic vote data assuming a specific data generating process with known parameters. Running the
  file `./data_generation/random_votes.py` will generate and save the random votes files necessary
  for this project. The functions there can also be imported and used directly.

- US data

    + The complete history of voting data as obtained from the VoteView project. Running the file
    `./data_generation/us_votes.py` will generate the files used in the project.

    + Cosponsoring sourced from James Fowler. Running the file `./data_generation/us_cosponsor.py`
    will generate the files used in the project.

- European Parliament votes as sourced from [ParlTrack](https://parltrack.org/). Running the file
    `./data_generation/eu_votes.py` will generate the files used in the project

## Models
