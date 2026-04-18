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
for i in range(len(lines)):
    s = lines[i]
    if '# Animation' in s:
        is_animation = True
    if is_animation and '####' in s:
        is_animation = False
    s = s.replace('show=True', 'show=False')
    s = s.replace('plt.show()', 'plt.close()')
    if is_animation:
        s = '\n'
    lines[i] = s

with open('example_temp.py', 'w') as f:
    for line in lines:
        f.write(line)

subprocess.run(shlex.split('python example_temp.py'))
os.remove('example_temp.py')

source_dir = Path('./')
dest_dir = Path('./example_data/output/')
for pattern in ('*.png', '*.html'):
    for file in source_dir.glob(pattern):
        file.rename(dest_dir / file.name)
for file in glob.glob("*.mp4") + glob.glob("*.html"):
    os.remove(file)


def images_are_close(img_path1, img_path2, tolerance=0):
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    if img1.size != img2.size:
        return False, -1

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    ref_diff = np.percentile(diff, 50)
    return ref_diff <= tolerance, ref_diff


reslist = []
difflist = []
pnglist = glob.glob("./example_data/output_expected/*.png")
for file in pnglist:
    expected = file
    output = file.replace('/output_expected/', '/output/')
    res, diff = images_are_close(output, expected, tolerance=2)
    reslist.append(res)
    difflist.append(diff)
filelist = np.array(pnglist)
reslist = np.array(reslist)
difflist = np.array(difflist)


def test_filematch():
    if np.all(reslist):
        print('All files matched.')
    else:
        print('Mismatched files:')
        print(filelist[~reslist])
    print('Differences:')
    print(difflist)
    assert np.all(reslist)


if __name__ == '__main__':
    test_filematch()
