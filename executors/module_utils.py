import sys
import os

root_project_dir = os.path.realpath(__file__)
for _ in range(2):
    root_project_dir = os.path.dirname(root_project_dir)
print('root project dir:', root_project_dir)
sys.path.append(root_project_dir)