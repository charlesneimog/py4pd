import pd
import os

def py4pdshow(img):
    if not os.path.exists(img):
        img = os.path.join(pd.get_patch_dir(), img)
    
    pd.show_image(img)

