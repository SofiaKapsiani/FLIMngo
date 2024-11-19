import os
from PIL import Image

# --- cropp_imgs.ipynb --- #
def colour_channel_imgs(img_dir, img):
    '''Load and open each colour channel image
    red: microtubules,
    green: prot. of interest
    blue: nucleus
    yellow: ER'''
    img_path = os.path.join(img_dir, img)
    return [Image.open(img_path+'_red.png'), Image.open(img_path+'_green.png'),
                Image.open(img_path+'_blue.png'), Image.open(img_path+'_yellow.png')]


