import random
import shutil
import os


def make_paths(base_path):
    os.mkdir(f"{base_path}\\test")
    os.mkdir(f"{base_path}\\test\\images")
    os.mkdir(f"{base_path}\\test\\annots")
    os.mkdir(f"{base_path}\\train")
    os.mkdir(f"{base_path}\\train\\images")
    os.mkdir(f"{base_path}\\train\\annots")


def copy_files(images_path, annots_path, base_path, ratio=0.1):
    images = os.listdir(images_path)
    random.shuffle(images)
    to_move = int(len(images) * ratio)
    print(to_move)
    i = 0
    for image in images:
        basename = os.path.splitext(image)[0]
        image_path = f"{images_path}\\{image}"
        xml_path = f"{annots_path}\\{basename}.xml"
        if i <= to_move:
            shutil.copy(image_path, f"{base_path}\\test\\images")
            shutil.copy(xml_path, f"{base_path}\\test\\annots")
            i += 1
        else:
            shutil.copy(image_path, f"{base_path}\\train\\images")
            shutil.copy(xml_path, f"{base_path}\\train\\annots")


def main():
    base_path = r"..\Dataset"
    images_path = r"..\Dataset\images"
    annots_path = r"..\Dataset\annotations"
    make_paths(base_path)
    copy_files(images_path, annots_path, base_path)


if __name__ == '__main__':
    main()