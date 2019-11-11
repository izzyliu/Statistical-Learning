import os
import glob
from wand.image import Image

load_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resized_dataset')
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
full_width = 80
full_height = 80
factors = [(0, 0.8), (0.1, 0.9), (0.2, 1)]
flop = [False, True]

for category in categories :
    counter = 0
    load_path = os.path.join(load_root, category, '*g')
    files = glob.glob(load_path)

    for file in files :
        with Image(filename=file) as img:
            img.resize(full_width, full_height)

            for is_flop in flop :
                if is_flop :
                    img.flop()

                for width_start, width_end in factors :
                    for height_start, height_end in factors :
                        counter += 1
                        save_name = category + str(counter) + '.jpg'
                        save_path = os.path.join(save_root, category, save_name)

                        with img.clone() as next_img :
                            next_img.crop(int(full_width*width_start), int(full_height*height_start),
                                            int(full_width*width_end), int(full_height*height_end))
                            next_img.format = 'jpg'
                            next_img.save(filename=save_path)
