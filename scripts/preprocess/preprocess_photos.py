from retinaface.pre_trained_models import get_model
from pathlib import Path
import cv2
import json


def prepare_photos(foto_path, save_path):
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()

    with open(r'variables.json', 'r') as file:
        d = file.read()
    d = json.loads(d)
    d['train_photo_path'] = save_path

    with open(r'variables.json', 'w') as file:
        file.write(json.dumps(d))

    counter = 1
    save_path = str(Path(save_path).joinpath('{}.jpg'))

    photos = Path(foto_path).glob('*.[jpg][jpeg][png]')

    for fle in photos:
        img = cv2.imread(str(fle))
        faces = model.predict_jsons(img)
        for bb in faces:
            x1, y1, x2, y2 = [int(i) for i in bb['bbox']]
            side = int(max(x2 - x1, y2 - y1) * 1.2) // 2
            x_c, y_c = [max((x1 + x2) // 2, side), max((y1 + y2) // 2, side)]
            new_image = cv2.resize(img[y_c - side:y_c + side, x_c - side:x_c + side], (512, 512))
            cv2.imwrite(save_path.format(counter), new_image)
            counter += 1
