# Use an NVIDIA CUDA with Ubuntu 22.04 as a parent image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Miniconda to manage Python environments and dependencies
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy the current directory contents into the container at /usr/src/app
#COPY . /usr/src/app

# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "test3", "/bin/bash", "-c"]

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
#CMD ["conda", "run", "--no-capture-output", "-n", "test3", "python", "main.py"]
