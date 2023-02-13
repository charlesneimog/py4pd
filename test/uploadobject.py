import os
import sys
import requests

version = '0.5.0'
response = requests.get("https://api.github.com/repos/charlesneimog/py4pd/releases/latest")
objectVersion = response.json()['tag_name']
repo = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

if version != objectVersion and repo == 'master':
    print('Version mismatch. Please update the object version in the uploadobject.py file.')

    
              

