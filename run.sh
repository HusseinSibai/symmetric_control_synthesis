export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/matlab/R2023a/bin/glnxa64

pip3 install --no-cache-dir -r symControlSynthesis/requirements.txt

python3 symControlSynthesis/main.py

