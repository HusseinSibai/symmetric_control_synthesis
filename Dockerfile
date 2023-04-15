# Use the specified MATLAB base image
FROM mathworks/matlab:r2023a

# Update system packages and install Python
USER root
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
COPY symControlSynthesis/requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy Python and Matlab code to the container
COPY IFAC20_ship_matlab/ /symmetric_control_synthesis/IFAC20_ship_matlab/
COPY symControlSynthesis/ /symmetric_control_synthesis/symControlSynthesis/

# Set the working directory
WORKDIR /symmetric_control_synthesis

USER 1001

# Command to run when the container starts
CMD ["python", "python_code/main.py"]
