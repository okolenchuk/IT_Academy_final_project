from retinaface.pre_trained_models import get_model
from pathlib import Path
import cv2
import json


def update_vars(var, value):
    with open(str(Path('IT_Academy_final_project').joinpath('variables.json')), 'r') as file:
        d = file.read()
    d = json.loads(d)
    d[var] = value

    with open(str(Path('IT_Academy_final_project').joinpath('variables.json')), 'w') as file:
        file.write(json.dumps(d))


def prepare_photos(foto_path, save_path):
    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()

    update_vars('train_photo_path', save_path)

    counter = 1

    photos = Path(foto_path).glob('*.[jpg][jpeg][png]')

    if not Path(save_path).exists():
        Path(save_path).mkdir()
    save_path = str(Path(save_path).joinpath('{}.jpg'))

    for fle in photos:
        img = cv2.imread(str(fle))
        faces = model.predict_jsons(img)
        for bb in faces:
            if len(bb['bbox']):
                x1, y1, x2, y2 = [int(i) for i in bb['bbox']]
                side = int(max(x2 - x1, y2 - y1) * 1.2) // 2
                x_c, y_c = [max((x1 + x2) // 2, side), max((y1 + y2) // 2, side)]
                new_image = cv2.resize(img[y_c - side:y_c + side, x_c - side:x_c + side], (512, 512))
                cv2.imwrite(save_path.format(counter), new_image)
                counter += 1
