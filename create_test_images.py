#!/usr/bin/env python3
"""
Lokal test için basit test resmi oluştur
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

def create_test_images():
    """Çeşitli test için örnek resimler oluştur"""
    
    test_dir = Path("./test_images_local")
    test_dir.mkdir(exist_ok=True)
    
    # 1. Gürültülü resim (Noise test)
    np.random.seed(42)
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    img.save(test_dir / "noisy_256x256.png")
    print(f"✓ Created: noisy_256x256.png")
    
    # 2. Düşük çözünürlüklü resim (Super-resolution test)
    img = Image.new('RGB', (128, 128), color='white')
    pixels = img.load()
    for i in range(128):
        for j in range(128):
            pixels[i, j] = (i*2, j*2, (i+j)%256)
    img.save(test_dir / "lowres_128x128.png")
    print(f"✓ Created: lowres_128x128.png")
    
    # 3. Bulanık resim (Deblurring test)
    img = Image.new('RGB', (256, 256), color='lightgray')
    pixels = img.load()
    for i in range(256):
        pixels[i, i] = (100, 150, 200)
    img.save(test_dir / "blurry_256x256.png")
    print(f"✓ Created: blurry_256x256.png")
    
    # 4. Karışık degradation
    base = np.random.randint(100, 180, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(base)
    img.save(test_dir / "mixed_degradation_256x256.png")
    print(f"✓ Created: mixed_degradation_256x256.png")
    
    print(f"\n✓ All test images created in: {test_dir}")
    return test_dir

if __name__ == "__main__":
    create_test_images()
