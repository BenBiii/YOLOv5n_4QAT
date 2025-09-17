# make_wider_val_txt.py
input_path = "C:/python_work/tftrain/workspace/YOLOv5n_FACE/widerface/val/label.txt"
output_path = "C:/python_work/tftrain/workspace/YOLOv5n_FACE/widerface/val/wider_val.txt"

with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if line.startswith('#'):
            image_path = line[2:]  # remove '# '
            f_out.write(image_path + '\n')
