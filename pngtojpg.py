import glob
import os
from PIL import Image

target_path       = "dataset\\train"

target_folders = os.listdir(target_path)
for target_folder in target_folders:

    target_dir = os.path.join(target_path, target_folder)
    print(target_dir)
    for file in glob.glob(target_dir + "/*.png"):
        file_name = os.path.basename(file)
        png_im = Image.open(file)
        rgb_im = png_im.convert('RGB')
        print(file_name.replace('.png','.jpg'))
        rgb_im.save(target_dir+'//'+file_name.replace('.png','.jpg'))
    # for file in glob.glob(target_name + "/*.png"):
    #     file_name = file.replace(target_path+'\\','')
