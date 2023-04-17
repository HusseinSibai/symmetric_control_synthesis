# Use the specified MATLAB base image
FROM gcr.io/ris-registry-shared/matlab 

# Update system packages and install Python
USER root

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/fs1/ris/application/matlab/2022b/bin/glnxa64

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3.8-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 # Install necessary Python packages
RUN python3 -m venv venv && \
    . venv/bin/activate
COPY symControlSynthesis/requirements.txt /tmp/
RUN venv/bin/activate && pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy Python and Matlab code to the container
COPY IFAC20_ship_matlab/ /symmetric_control_synthesis/IFAC20_ship_matlab/
COPY symControlSynthesis/ /symmetric_control_synthesis/symControlSynthesis/
COPY run.sh /symmetric_control_synthesis/run.sh

# Set the working directory
WORKDIR /symmetric_control_synthesis


USER 1001

# Command to run when the container starts
#CMD ["python", "python_code/main.py"]
