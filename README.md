# BinaPs


## Licensing and authorship

For BinaPs information, please check the [BinaPs Code Authorship Disclaimer](DISCLAIMER) located at DISCLAIMER.

All code elsewhere is written by Bernardo C. Rodrigues and licensed under GPL-3.0-or-later. Please check COPYING for the
complete license.

## Code organization

- binaps: original BinaPs implementation
- environment: a plug n' play environment for running BinaPs and/or extending it.
- notebooks: Jupyter notebooks that explore the BinaPs capabilities
- lib: helper python modules

## Plug n' Play Environment

All BinaPs dependencies along with other data science packages are organized in the docker-compose under environment/. This environment can be used as a stand-alone Jupyter Notebook or be connected to a external editor such as VSCode (with the Jupyter support installed).

### CUDA Support

If you have a CUDA capable GPU and with to use it, you need to install [NVIDIA's container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Follow the official installation guide.

### Building the Environment

        cd environment
        docker compose build

This will place a *environment_notebook-cuda* image at your local docker run time

### Stand-alone Jupyter notebook

First, provide some host information so the containers can be instantiated correctly. This information can be supplied
in environment variable or in a file placed, along with the docker-compose.yml, at environment/. See the following
example:

        # environment/.env

        USER=foo
        UID=1000
        GID=1000
        BINAPS_ROOT_DIR=/home/foo/binaps

The environment can be ran as follows:

        # CUDA support
        docker compose run --rm notebook-cuda

        # No CUDA support
        docker compose run --rm notebook

Access Jupyter notebook through the browser with URL provided in the terminal.

### Using as an IPython Kernel

Install the example *environment/kernel.json* at your local IPython kernel directory. Substitute the placeholder 
[BINAPS_ROOT_DIR] inside kernel.json with the appropriate value.

        mkdir ~/.local/share/jupyter/kernels/binaps/
        cp environment/kernel.json ~/.local/share/jupyter/kernels/binaps/kernel.json

### Running unit tests

To run the unit test suit, make sure you've built the environment and simply run the script at `ci/test`. It will run
pytest inside the provided environment.