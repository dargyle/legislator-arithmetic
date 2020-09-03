# legislator-arithmetic A neural network approach to ideal point estimation

## Data acquisition and generation

There are several datasets used in this project, all of which can be generated programmatically
(assuming none of the links have broken in the interim...)

- Synthetic vote data assuming a specific data generating process with known parameters. Running the
  file ./data_generation/random_votes.py will generate and save the random votes files necessary for
  this project. The functions there can also be imported and used directly.

- US data

    + The complete history of voting data as obtained from the VoteView project. Running the file
    ./data_generation/us_votes.py will generate the files used in the project.

    + Cosponsoring sourced from James Fowler. Running the file ./data_generation/us_cosponsor.py
    will generate the files used in the project.

- European Parliament votes as sourced from ParlTrack. Running the file
    ./data_generation/eu_votes.py will generate the files used in the project

## Models
