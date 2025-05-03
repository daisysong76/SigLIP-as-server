import os
import numpy as np
import pickle
import tempfile
import shutil
from PIL import Image
import pytest

# No need to import variables from generate_embeddings

def setup_test_images(tmpdir, n=3):
    os.makedirs(tmpdir, exist_ok=True)
    paths = []
    for i in range(n):
        img = Image.new('RGB', (32, 32), color=(i*40, i*40, i*40))
        path = os.path.join(tmpdir, f"test_{i}.jpg")
        img.save(path)
        paths.append(path)
    return paths

def test_generate_embeddings(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    try:
        import generate_embeddings
        # Setup test images
        monkeypatch.setattr('generate_embeddings.IMAGE_DIR', tmpdir)
        monkeypatch.setattr('generate_embeddings.OUTPUT_METADATA', os.path.join(tmpdir, 'metadata.pkl'))
        setup_test_images(tmpdir, n=2)
        # Run main logic
        generate_embeddings.main()
        # Check metadata file
        assert os.path.exists(generate_embeddings.OUTPUT_METADATA)
        with open(generate_embeddings.OUTPUT_METADATA, 'rb') as f:
            meta = pickle.load(f)
        assert len(meta) == 2
        for m in meta:
            assert 'clip_embedding' in m and 'siglip_b_embedding' in m and 'siglip_l_embedding' in m
    finally:
        shutil.rmtree(tmpdir) 