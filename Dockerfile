# Use the specified MATLAB base image
FROM gnuoctave/octave:6.2.0

# Update system packages and install Python
USER root

RUN apt-get update && \
    apt-get install -y python3.9 python3-pip python3.9-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 # Install necessary Python packages
RUN python3.9 -m venv venv && \
    . venv/bin/activate
COPY symControlSynthesis/requirements.txt /tmp/
RUN . venv/bin/activate && python3.9 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy Python and Matlab code to the container
COPY IFAC20_ship_matlab/ /symmetric_control_synthesis/IFAC20_ship_matlab/
COPY symControlSynthesis/ /symmetric_control_synthesis/symControlSynthesis/
COPY run.sh /symmetric_control_synthesis/run.sh

# Set the working directory
WORKDIR /symmetric_control_synthesis


USER 1001

# Command to run when the container starts
#CMD ["python", "python_code/main.py"]
