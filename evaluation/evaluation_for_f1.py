# ==============================Change Path=======================================
import os, sys

abs_path = os.path.abspath(sys.argv[0])
dir_path = abs_path.replace("/evaluation/evaluation_for_f1.py", "")
sys.path.append(dir_path)
# ================================================================================

import json 

from seq2seq.metrics.lc_quad.lc_quad_f1 import compute_f1_metric



def get_prediction_and_reference_frm_file(filename):
