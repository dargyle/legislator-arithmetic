# legislator-arithmetic A neural network approach to ideal point estimation

## Installation instructions

### Code download from github

Install the legislator-arithmetic package from github (for example: `git clone
https://github.com/dargyle/legislator-arithmetic.git`)

### Data location

The location for the data storage is set in the `constants.py` file, by default it's set to
`/data/leg_math/` in the primary user folder

### Conda enviroment

The `environment.yml` file in the main directory contains a configuration file that will generate a
conda enviroment called `leg_math_new` containing all the packages necessary to run the models. It
requires a pre-existing installation of of the conda package manager, either through
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or full
[Anaconda](https://docs.anaconda.com/anaconda/install/).

Assuming a working conda instance navigate to the package folder and install the `leg_math_new`
environment using this command:

    conda env create -f environment.yml

For additional help with conda and environments see the
[conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### R (optional)

Scripts for replication of some results rely on the rpy2 package to connect with R. This requires a
working R installation that has the `pscl` and `wnominate` packages installed. R is not necessary
to use the NN-NOMINATE code.

### lzip (optional)

[ParlTrack](https://parltrack.org/) (the EU vote source) compresses their files using lzip
(<https://www.nongnu.org/lzip/>). This needs to be installed and available via the command line to
acquire the EU vote data.

## Data acquisition and generation

There are several datasets used in this project, all of which can be generated programmatically
(assuming none of the links have broken in the interim...)

-   Synthetic vote data assuming a specific data generating process with known parameters. Running the
    file `./data_generation/synthetic_votes.py` will generate and save the random votes files necessary
    for this project. The functions there can also be imported and used directly if desired.

-   US data

    -   The complete history of voting data as obtained from the VoteView project. Running the file
        `./data_generation/us_votes.py` will generate the files used in the project.

    -   Cosponsoring sourced from James Fowler. Running the file `./data_generation/us_cosponsor.py`
        will generate the files used in the project.

-   European Parliament votes as sourced from [ParlTrack](https://parltrack.org/). Running the file
      `./data_generation/eu_votes.py` will generate the files used in the project

## Keras Models
-   test_models.py: A generic script designed to test the models

# Applications
-   synthetic_data_eval.py: evaluates the model on randomly generated votes with known parameters and
    compares performance to the wnominate package in R. (An R install is reqired to run this file, see above.)
