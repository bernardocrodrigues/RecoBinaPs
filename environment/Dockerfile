FROM nvidia/cuda:12.0.0-base-ubuntu22.04

## Install Python
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-pip

## Install basic data science libraries
# Required by the BinaPs implementation
RUN pip install numpy \
                scipy \
                pandas \
                torch \
                torchvision \
                torchaudio

# Install Jupyter and IPython support
RUN pip install jupyterlab \
                ipykernel \
                ipython
RUN python3 -m ipykernel install --name=binaps

## Install R
# To enable us to run binaps/Data/Synthetic_data/generate_toy.R
RUN DEBIAN_FRONTEND=noninteractive apt -y install r-base

# Install extension dependencies
RUN pip install rich \
                surprise \
                numba \
                pydantic

# Data visualization dependencies
RUN pip install matplotlib \
                plotly \
                tabulate \
                plotext \
                statsmodels \
                kaleido
RUN DEBIAN_FRONTEND=noninteractive apt -y install msttcorefonts

# Unit test tooling
RUN pip install pytest \
                coverage \
                nbmake
RUN pip install coverage

# Install Pedro's implementation dependencies
RUN pip install scikit-learn

# Install SonarScanner
RUN apt install unzip
RUN wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-5.0.1.3006-linux.zip?_gl=1*13xosil*_gcl_au*NDg4MTYxNTg4LjE3MDY2NDYyMTQ.*_ga*ODk2ODA3NDk4LjE3MDY2NDYyMTQ.*_ga_9JZ0GZ5TC6*MTcwNjY0NjIxNC4xLjEuMTcwNjY1MzQyOC4zNS4wLjA. -O sonar-scanner-cli-5.0.1.3006-linux.zip
RUN unzip sonar-scanner-cli-5.0.1.3006-linux.zip
RUN mv sonar-scanner-5.0.1.3006-linux /opt/sonar-scanner
RUN ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner
RUN rm sonar-scanner-cli-5.0.1.3006-linux.zip

# Install Arrow to fix Pandas requirements warning
RUN pip install pyarrow

# Install CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-2