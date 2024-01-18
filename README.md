# Symmetric Control Synthesis

## Setup

### On Apple M2 Pro

- Follow instructions for installing [Amazon Corretto 8](https://www.mathworks.com/support/requirements/apple-silicon.html)
  - You may have to [remove existing Java versions](https://www.java.com/en/download/help/mac_uninstall_java.html)
- Create a  MathWorks account and get an [academic license](https://www.mathworks.com/products/matlab-campus.html)
- Download the [latest Matlab version](https://www.mathworks.com/downloads) for Apple silicon
- Follow the installer's instruction
  - Check 7 additional packages:
    - Communications Toolbox
    - Computer Vision Toolbox
    - Control System Toolbox
    - DSP System Toolbox
    - Image Processing Toolbox
    - Signal Processing Toolbox
    - Statistics and Machine Learning Toolbox
- Include IFAC20 in your Download folder (temporary setup)
- Python version: anything above 3.9 should work, be sure to use the base python installation and not use python through anaconda
- Python packages: rtree, matlab, matlabengine, polytope, multiprocess, shared_memory_dict, qpsolvers\[clarabel\]

## Running the experiments

### Replicating the paper's results

- Using symControlSynthesis as a working directory start the launcher with
```python3 mainLauncher.py```
You will be prompted to which tests you want to replicate and give a folder name to store the results
- Some results might not exactly match the numbers given in the paper. The reasons for this are:
  - The way neighboring cells are computed each iteration of the synthesis depends on the order in which controllable states are discovered, which varies with poll assignment and parallelization
  - The results for the benchmark of the largest experiments in the paper were computed on a slightly different version of the synthesis