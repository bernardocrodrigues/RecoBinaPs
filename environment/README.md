# Environment

This directory contains the Dockerfile and docker-compose files used to create
the environments for running the code. This environment can be used as a
stand-alone Jupyter Notebook or be connected to a external editor such as VSCode
(with the Jupyter support installed).

## Components

### Dockerfile

The Dockerfile defines the base image and the packages that will be installed
in the image.

### Docker Compose Files

The `base.yml` and `base-cuda.yml` files define the base services for CPU and
GPU environments respectively. The `notebook.yml` file defines the service for
running Jupyter notebooks.

To run a service, use the `docker-compose` command. For example, to run the
`notebook` service:

## Setup

Follow the instructions below to build and run the environment.

1. [**Optional**] Install 
   [NVIDIA's container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
   This is needed if you have a CUDA capable GPU and wish to use it with the
   environment.
2. Build the environment image.
   ```bash
   # Build the base image
   docker compose -f base.yml build
   # Build the base image with CUDA support
   docker compose -f base-cuda.yml build
   # Build the notebook image
   docker compose -f notebook-cuda.yml build

   ```
3. Provide the environment variables
   
   Since we are using docker-compose to handle the environment parametrization,
   first, provide some host information so the containers can be instantiated
   correctly. This information can be supplied as environment variables or in a
   file placed, along with the docker compose yml's, at `environment/`. 

   ```bash
   # Example contents of environment/.env

   USER=foo # The user that will run the environment
   UID=1000 # The user's UID
   GID=1000 # The user's GID
   SOURCE_ROOT_DIR=/home/path/to/project # The root directory of the repository
   ```


4. Use the environment
   - **as a stand-alone Jupyter notebook**. 
      
      The environment can be ran as follows:

      ```bash
      cd environment

      # CUDA support
      docker compose -f notebook.yml run --rm notebook-cuda

      # No CUDA support
      docker compose -f notebook.yml run --rm notebook
      ```
      
      Access Jupyter notebook through the browser with URL provided in the
      terminal.

    - **as a remote Jupyter server**. Perform the same setup described in 4.1
      except that after launching the Jupyer server, you should connect to it
      through the desired client (e.g. VSCode) using the URL and the token
      'something_secret'.
   
    - **as an IPython Kernel**. Install the example *environment/kernel.json* at
      your local IPython kernel directory. Substitute the placeholder
      \[SOURCE_ROOT_DIR\] inside kernel.json with the appropriate value.
       ```bash
       mkdir ~/.local/share/jupyter/kernels/recommender/
       cp environment/kernel.json ~/.local/share/jupyter/kernels/recommender/kernel.json
       ```
      Then, you can use the environment as a kernel in your favorite editor.