import pd
import os

def py4pdshow(img):
    if not os.path.exists(img):
        img = os.path.join(pd.home(), img)
    
    pd.show(img)

