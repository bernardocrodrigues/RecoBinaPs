# Improving Recommender Systems
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=recommender-systems&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=recommender-systems)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=recommender-systems&metric=bugs)](https://sonarcloud.io/summary/new_code?id=recommender-systems)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=recommender-systems&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=recommender-systems)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=recommender-systems&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=recommender-systems)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=recommender-systems&metric=coverage)](https://sonarcloud.io/summary/new_code?id=recommender-systems)

Supporting code of my ongoing Master's thesis, titled "Improving Recommender Systems through 
Pattern Mining".

## Code Organization

Code in this repository is composed of Python modules, notebooks, scripts and other files. 

### Modules
- **datasets**: Dataset loading and preprocessing.
- **evaluation**: Evaluation metrics.
- **pattern_mining**:  Pattern mining algorithms. (e.g. GreConD)
- **recommenders**: Multiple [Surprise](https://surprise.readthedocs.io/en/stable/) based
  Recommender Systems implementations.

### Notebooks

- **demos**: Jupyter notebooks used to demonstrate the usage of the modules and concepts.
- **experiments**: Jupyter notebooks used to generate the results of the experiments.

### Scripts and Misc

- **ci**: Continuous integration scripts.
- **environment**: Docker environment for running anything in this repository.
- **scripts**: Support scripts for other tasks.
- **tests**: Unit tests.

## Unit Tests

1. Build the environment image as described in the [Environment Setup](environment/README.md#environment-setup) section.
2. Run the unit tests.
   ```bash
   cd ci
   ./test -f # Run the complete test suite, including the notebooks and coverage analysis
   ./test -h # See help for more options

   ```
   This will run the unit tests inside the environment and output the results to the terminal.

## Running the Notebooks

1. Build the environment image as described in the [Environment Setup](environment/README.md#environment-setup) section.
2. Start the Jupyter Server

    ```bash
    cd environment

    # Run the server
    docker compose -f notebook.yml run --build --rm notebook

    # Or run with CUDA support
    docker compose -f notebook-cuda.yml run --build --rm notebook-cuda
    ```
3. Either open the link printed in the terminal or connect to the server using the VSCode Jupyter     extension. Check the [Environment Setup](environment/README.md#environment-setup) section for more details.
4. Run the notebooks in the `demos` and `experiments` directories.

## Licensing and authorship

For BinaPs information, please check the [BinaPs Code Authorship Disclaimer](DISCLAIMER) located at DISCLAIMER.

All code elsewhere is written by Bernardo C. Rodrigues and licensed under GPL-3.0-or-later. Please check COPYING for the
complete license.