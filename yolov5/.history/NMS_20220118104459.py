# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import glob
import os

def convert_yolov5_preds(valid_files_dir, labels_dir, out_file):
    valid_files = glob.glob(valid_files_dir + '*.jpg' )
    print('Total image files: {}'.format(len(valid_files)))
    files = glob.glob(labels_dir + '*.txt')
    print('Total labels files: {}'.format(len(files)))
    valid_ids = [os.path.basename(f)[:-4] for f in valid_files]
    out = open(out_file, 'w')
    out.write('image_id,label,conf,x1,x2,y1,y2\n')
    fixes = 0
    for f in files:
        image_id = os.path.basename(f)[:-4]
        in1 = open(f, 'r')
        lines = in1.readlines()
        in1.close()
        valid_ids.remove(image_id)
        for line in lines:
            arr = line.strip().split(' ')
            class_id = arr[0]
            x = float(arr[1])
            y = float(arr[2])
            w = float(arr[3])
            h = float(arr[4])
            x1 = x - (w / 2)
            x2 = x + (w / 2)
            y1 = y - (h / 2)
            y2 = y + (h / 2)
            if x1 < 0:
                x1 = 0
            if x2 > 1:
                fixes += 1
                x2 = 1
            if y1 < 0:
                fixes += 1
                y1 = 0
            if y2 > 1:
                fixes += 1
                y2 = 1
            conf = arr[5]
            pred_str = '{},{},{},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(image_id, str(class_id), conf, x1, x2, y1, y2)
            out.write(pred_str)

    print(len(valid_ids))

    # Output empty IDs
    for image_id in list(valid_ids):
        out.write('{},,,,,,\n'.format(image_id))

    out.close()
    print('Fixes: {}'.format(fixes))
    print('Result was written in: {}'.format(out_file))


if __name__ == '__main__':
    # Location of images
    valid_files_dir = 'Maskdata/test/images'
    # Location of yolo v5 predictions
    labels_dir = 'Maskdata/test/labels'
    # CSF-file to store results
    out_file = './text_results.csv'
    convert_yolov5_preds(valid_files_dir, labels_dir, out_file)