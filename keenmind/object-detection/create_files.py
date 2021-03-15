import os
import sys
import random


root_dir = sys.argv[1]


def main(root_dir):

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    split = 0.7

    images = os.listdir(images_dir)
    N = len(images)
    split_val = int(split*N)
    random.shuffle(images)
    
    train_imgs = images[:split_val]
    val_imgs = images[split_val:]

    full_train_imgs = [os.path.join(images_dir, img) for img in train_imgs]
    full_val_imgs = [os.path.join(images_dir, img) for img in val_imgs]

    train_path = os.path.join(root_dir, 'train.txt')
    valid_path = os.path.join(root_dir, 'valid.txt')

    with open(train_path, 'w') as f:
        for item in full_train_imgs:
            f.write("%s\n" % item)
    
    with open(valid_path, 'w') as g:
        for item in full_val_imgs:
            g.write("%s\n" % item)

if __name__ == "__main__":
    main(root_dir)
