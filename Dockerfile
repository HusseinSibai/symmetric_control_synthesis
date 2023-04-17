# Use the specified MATLAB base image
FROM gcr.io/ris-registry-shared/matlab

# Update system packages and install Python
USER root

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python and Matlab code to the container
COPY IFAC20_ship_matlab/ /symmetric_control_synthesis/IFAC20_ship_matlab/
COPY symControlSynthesis/ /symmetric_control_synthesis/symControlSynthesis/

# Set the working directory
WORKDIR /symmetric_control_synthesis


