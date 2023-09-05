# Improving Recommender Systems

Supporting code of my ongoing Master's thesis, titled "Improving Recommender Systems through 
Pattern Mining".

## Code Organization

Code in this repository is composed of Python modules, notebooks, scripts and other files. 

### Modules

- **binaps**: BinaPs implementation. Contains the original BinaPs code, a lib friendly refactor and 
  a wrapper for using the original code as a lib.
- **datasets**: Dataset loading and preprocessing.
- **evaluation**: Evaluation metrics.
- **fca**: Any implementation related to Formal Concept Analysis such as the GreConD algorithm.
- **recommenders**: Multiple [Surprise](https://surprise.readthedocs.io/en/stable/) based
  Recommender Systems implementations.

### Notebooks

- **demos**: Jupyter notebooks used to demonstrate the usage of the modules and concepts.
- **results**: Jupyter notebooks used to generate the results of the experiments.

### Scripts and Misc

- **ci**: Continuous integration scripts.
- **environment**: Docker environment for running anything in this repository.
- **scripts**: Support scripts for other tasks.
- **tests**: Unit tests.

## Environment

All dependencies and environment setup are defined in the Docker image at `environment/`. This
environment can be used as a stand-alone Jupyter Notebook or be connected to a external editor such
as VSCode (with the Jupyter support installed).

Follow the instructions below to build and run the environment.


1. [**Optional**] Install 
   [NVIDIA's container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
   This is needed if you have a CUDA capable GPU and wish to use it with the environment.
2. Build the environment image.
   ```bash
   cd environment
   docker compose build
   ```
   This will place a *environment_notebook-cuda* image at your local docker run time
3. Use the environment
   1. **as a stand-alone Jupyter notebook**. Since we are using docker-compose to handle the
      environment parametrization, first, provide some host information so the containers can be
      instantiated correctly. This information can be supplied as environment variables or in a
      file placed, along with the docker-compose.yml, at `environment/`. 
      ```bash
      # environment/.env

      USER=foo
      UID=1000
      GID=1000
      BINAPS_ROOT_DIR=/home/foo/binaps
      ```
      
      The environment can be ran as follows:

      ```bash
      # CUDA support
      docker compose run --rm notebook-cuda

      # No CUDA support
      docker compose run --rm notebook
      ```
      
      Access Jupyter notebook through the browser with URL provided in the terminal.

    2. **as a remote Jupyter server**. Perform the same setup described in 3.1 except that after
       launching the Jupyer server, you should connect to it through the desired client
       (e.g. VSCode) using the URL and the token 'something_secret'.
   
    3. **as an IPython Kernel**. Install the example *environment/kernel.json* at your local IPython
       kernel directory. Substitute the placeholder [BINAPS_ROOT_DIR] inside kernel.json with the
       appropriate value.
       ```bash
       mkdir ~/.local/share/jupyter/kernels/binaps/
       cp environment/kernel.json ~/.local/share/jupyter/kernels/binaps/kernel.json
       ```

## Unit Tests

1. Build the environment image as described in the previous section.
2. Run the unit tests.
   ```bash
   cd ci
   ./test
   ```
   This will run the unit tests inside the environment and output the results to the terminal.

## Licensing and authorship

For BinaPs information, please check the [BinaPs Code Authorship Disclaimer](DISCLAIMER) located at DISCLAIMER.

All code elsewhere is written by Bernardo C. Rodrigues and licensed under GPL-3.0-or-later. Please check COPYING for the
complete license.