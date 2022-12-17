from retinaface.pre_trained_models import get_model
from pathlib import Path
import cv2


def prepare_photos(foto_path, save_path):
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()

    with open(r'variables.json', 'a') as file:
        d = file.read()
        d['train_photo_path'] = save_path

    counter = 1

    for fle in Path(foto_path).glob('*.jpg'):
        img = cv2.imread(str(fle))
        faces = model.predict_jsons(img)
        for bb in faces:
            x1, y1, x2, y2 = [int(i) for i in bb['bbox']]
            side = max(x2 - x1, y2 - y1) // 2
            x_c, y_c = [max((x1 + x2) // 2, side), max((y1 + y2) // 2, side)]
            new_image = cv2.resize(img[y_c - side:y_c + side, x_c - side:x_c + side], (512, 512))
            cv2.imwrite(save_path.format(counter), new_image)
            counter += 1
