"""
Run every step in order. Regenerates everything in data/ (except vegf_parameters.csv)
and everything in figures/.
"""

import runpy

for script in ["vegf_step1.py", "vegf_step2.py", "vegf_step3.py", "vegf_step4.py",
               "generate_training_data.py", "vegf_step5.py"]:
    print(f"\n========== {script} ==========")
    runpy.run_path(script, run_name="__main__")
