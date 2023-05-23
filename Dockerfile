# Official Ubuntu base image with Python 3.10
FROM ubuntu:22.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages and run setup.py
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.9 python3-pip python3.9-dev curl
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -e .
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Get the latest available model
RUN curl -L -o models/skinvestigator_nano_40MB_91_38_acc.h5 https://github.com/Thomasbehan/SkinVestigatorAI/releases/download/0.0.3/skinvestigator_nano_40MB_91_38_acc.h5

# Add a new user to avoid running the application as root
RUN useradd -ms /bin/bash appuser
USER appuser

# Make port 6543 available to the world outside this container and 6006 for TensorBoard
EXPOSE 6543 6006

# Define environment variable
ENV NAME World

# Ensure the pserve command is in the PATH
ENV PATH="/app/.local/bin:${PATH}"

# Run command when the container launches
CMD ["pserve", "development.ini", "--reload"]