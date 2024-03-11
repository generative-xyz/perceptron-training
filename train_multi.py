import os
import subprocess

input_path = 'data/cryptopunks'
output_path = 'models/cryptopunks'

for dir in next(os.walk(input_path))[1]:
  data_input = os.path.join(input_path, dir)
  model_output = os.path.join(output_path, dir) + '.json'

  cmd = ["python3", "training_user.py", "-c", "config.json", "-d", data_input, "-o", model_output]
  print(subprocess.list2cmdline(cmd))
  subprocess.run(cmd)
