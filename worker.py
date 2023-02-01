#!/home/ethan.marx/miniconda3/envs/hp-search/bin/python
print("hey")
import os, re, subprocess, time
from pathlib import Path

import ray
print("imported ray")

pwd = Path(".").resolve()

while not os.path.exists(pwd / 'condor.out'):
    time.sleep(1)

machine = None
for num in range(100):
    if machine:
        break
    with open(pwd / 'condor.out') as f:
        for line in f:
            match = re.match('.*\'(.*):6379\'.*', line)
            print(match)
            if match:
                machine = match.groups()[0]
                print(machine)
                break
    time.sleep(10)
else:
    raise RuntimeError("No match found")

print(f"Connecting to ray on {machine}")

#ray.init(address=f"ray://{machine}:10001", _temp_dir = "/home/ethan.marx/ray/")
subprocess.run(f'/home/ethan.marx/miniconda3/envs/hp-search/bin/ray start --block --num-cpus 2 --address={machine}:6379 --temp-dir /home/ethan.marx/ray/', shell=True)

print("Connected")
