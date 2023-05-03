export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/fs1/ris/application/matlab/2022b/bin/glnxa64

. /venv/bin/activate

python3.9 -m pip install --no-cache-dir -r symControlSynthesis/matlab_req.txt

python3.9 symControlSynthesis/main.py

