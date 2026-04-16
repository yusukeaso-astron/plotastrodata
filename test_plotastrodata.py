import glob
import numpy as np
import os
import shlex
import subprocess
from pathlib import Path
from PIL import Image


with open('example.py', 'r') as f:
    lines = f.readlines()
is_animation = False
is_html = False
for i in range(len(lines)):
    s = lines[i]
    if '# Animation' in s:
        is_animation = True
    if '# 3D html' in s:
        is_html = True
    if is_animation and '####' in s:
        is_animation = False
    if is_html and '####' in s:
        is_html = False
    s = s.replace('show=True', 'show=False')
    s = s.replace('plt.show()', 'plt.close()')
    if is_animation or is_html:
        s = '\n'
    lines[i] = s

with open('example_temp.py', 'w') as f:
    for line in lines:
        f.write(line)

subprocess.run(shlex.split('python example_temp.py'))
os.remove('example_temp.py')

source_dir = Path('./')
dest_dir = Path('./example_data/output/')
for file in source_dir.glob('*.png'):
    file.rename(dest_dir / file.name)
for file in glob.glob("*.mp4") + glob.glob("*.html"):
    os.remove(file)


def images_are_close(img_path1, img_path2, tolerance=0):
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    if img1.size != img2.size:
        return False

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    max_diff = np.percentile(diff, 99)
    return max_diff <= tolerance


reslist = []
pnglist = glob.glob("./example_data/output_expected/*.png")
for file in pnglist:
    output = file
    expected = file.replace('/output_expected/', '/output/')
    res = images_are_close(output, expected, tolerance=2)
    reslist.append(res)
reslist = np.array(reslist)
pnglist = np.array(pnglist)


def test_pngfiles():
    if np.any(reslist):
        print(pnglist[reslist])
    assert np.any(reslist)
