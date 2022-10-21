To process an image pair:
1. Put stereo.py and two images (left.png and right.png) in the same directory.
2. From a terminal, activate the CS6476 virtual environment: ~$ conda activate CS6476
3. From the stereo.py directory, run stereo.py file: ~$ python3 stereo.py
4. You will be asked to enter ndisp value.
5. You will see outputs "Processing SSD left to right", "Processing SSD right to left" and "Processing DP", indicating which method is
being processed. When you see "All images completed", all the 3 outputs will be saved to the same directory.

To compare student outputs with ground truth:
1. Put metrics.py and two images (student.png and ground.png) in the same directory.
2. From a terminal, activate the CS6476 virtual environment: ~$ conda activate CS6476
3. From the metrics.py directory, run metrics.py file: ~$ python3 metrics.py
4. You will be asked to enter ndisp value.
5. You will see outputs "Comparing histogram", "The histogram correlation is xxx", "Comparing normalized averaged difference", "The normalized averaged difference is xxx",
which are the two metrics used for discussion in the report.