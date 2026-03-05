import os
import cv2
import torch
import numpy as np
import png

from train import Net

IMAGE_DIR = './figures'
MODEL_PATH = './models/model_UNet.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORMAP = [
    '#000000',
    '#0080FF',
    '#80FF80',
    '#FF8000',
    '#FF0000',
]

def hex2rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


net = Net().to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

palette = [hex2rgb(c) for c in COLORMAP]

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.lower().endswith(('.jpg',)):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    out_path = os.path.join(
        IMAGE_DIR,
        fname.replace('.jpg', '.png')
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 128.0 - 1.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = net(img)[0]
        pred = torch.argmax(logits, dim=0).cpu().numpy().astype('uint8')

    w = png.Writer(pred.shape[1], pred.shape[0], palette=palette, bitdepth=4)
    with open(out_path, 'wb') as f:
        w.write(f, pred)

    print(f'Saved: {out_path}')