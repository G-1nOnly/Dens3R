# Copyright (C) 2025-present Alibaba Group. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# COLMAP database utils
# --------------------------------------------------------
import os
import io
import sys
import h5py
import shutil
import pycolmap
import subprocess
import contextlib
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from mast3r.colmap_utils.database import COLMAPDatabase
from typing import Optional, List, Dict, Any

def create_empty_db(database_path: Path):
    if database_path.exists():
        # logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    # logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir: Path,
                  database_path: Path,
                  camera_mode: pycolmap.CameraMode,
                  image_list: Optional[List[str]] = None,
                  options: Optional[Dict[str, Any]] = None):
    # logger.info('Importing images into the database...')
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)
        
def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
    if return_uncertainty:
        return p, uncertainty
    return p

def import_features(image_ids: Dict[str, int],
                    database_path: Path,
                    features_path: Path):
    # logger.info('Importing features into the database...')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()
    
def find_pair(hfile: h5py.File, name0: str, name1: str):
    def names_to_pair(name0, name1, separator='/'):
        return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))
    def names_to_pair_old(name0, name1):
        return names_to_pair(name0, name1, separator='_')
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')
    
def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores
    
def import_matches(image_ids: Dict[str, int],
                   database_path: Path,
                   pairs_path: Path,
                   matches_path: Path,
                   min_match_score: Optional[float] = None,
                   skip_geometric_verification: bool = False):
    # logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()
    
class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            # if exc_type is not None:
                # logger.error('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()
    
def estimation_and_geometric_verification(database_path: Path,
                                          pairs_path: Path,
                                          verbose: bool = False):
    # logger.info('Performing geometric verification of the matches...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(
                database_path, pairs_path)
            
def run_reconstruction(sfm_dir: Path,
                       database_path: Path,
                       image_dir: Path,
                       verbose: bool = False,
                       options: Optional[Dict[str, Any]] = None,
                       ) -> pycolmap.Reconstruction:
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    if options is None:
        options = {}
    options = {'num_threads': min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options)

    if len(reconstructions) == 0:
        return None

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(
            str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]

def glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path):
    print("running GLOMAP mapping...")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, glomap_bin)
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')
    ouput_recon = pycolmap.Reconstruction(os.path.join(recon_path, '0'))
        
    return ouput_recon

def run_triangulation(model_path: Path,
                      database_path: Path,
                      image_dir: Path,
                      reference_model: pycolmap.Reconstruction,
                      verbose: bool = False,
                      options: Optional[Dict[str, Any]] = None,
                      ) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = {}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, image_dir, model_path,
                options=options)
    return reconstruction

def create_db_from_model(reconstruction: pycolmap.Reconstruction,
                         database_path: Path) -> Dict[str, int]:
    if database_path.exists():
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(
            camera.model.value, camera.width, camera.height, camera.params,
            camera_id=camera.camera_id, prior_focal_length=True)

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in reconstruction.images.items()}
    
def import_keypts(database_path: Path,
                  imgs_keypts_dict: dict,
                  img_name_dict: dict):
    db = COLMAPDatabase.connect(database_path)
    
    for img_name, img_keypts_dict in tqdm(imgs_keypts_dict.items(), desc="Importing keypts"):
        key_pts = np.array(list(img_keypts_dict.keys())).astype(np.float64)
        key_pts += 0.5  # COLMAP origin
        img_id = img_name_dict[img_name]
        if (len(key_pts.shape) != 2) or (key_pts.shape[1] not in [2, 4, 6]):
            db.execute("DELETE FROM images WHERE name=?", (img_name,))
            continue
        db.add_keypoints(img_id, key_pts)

    db.commit()
    db.close()
    
def import_mts(database_path: Path,
               matches_dict: dict,
               skip_geometric_verification: bool = False):
    db = COLMAPDatabase.connect(database_path)
    matched = set()
    for (id0, id1), matches_pts_idx in tqdm(matches_dict.items(), desc="Importing matches"):
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        if len(matches_pts_idx.shape) != 2:
            continue
        db.add_matches(id0, id1, matches_pts_idx)
        matched |= {(id0, id1), (id1, id0)}
        
        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches_pts_idx)

    db.commit()
    db.close()
