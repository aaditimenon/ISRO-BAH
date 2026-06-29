import numpy as np
from pathlib import Path
import pipeline.config as cfg

def npz_path(tic_id, sector, kind='raw'):
    """Return Path object for a given star and data kind."""
    if kind == 'raw':
        parent_dir = cfg.RAW_DIR
    elif kind == 'preprocessed':
        parent_dir = cfg.PREP_DIR
    else:
        parent_dir = cfg.OUTPUTS_DIR / kind
    
    parent_dir.mkdir(parents=True, exist_ok=True)
    return parent_dir / f"{tic_id}_s{sector}.npz"

def load_lc_npz(tic_id, sector, kind='raw'):
    """Load lightcurve data from npz file and return as a dict-like object."""
    path = npz_path(tic_id, sector, kind=kind)
    if not path.exists():
        raise FileNotFoundError(f"No {kind} npz file found for TIC {tic_id} Sector {sector} at {path}")
    
    # Using a context manager so we don't leak open file descriptors
    with np.load(str(path), allow_pickle=True) as data:
        result = {}
        for key in data.files:
            val = data[key]
            # If it's a 0-d array containing object/dict (like meta), extract it
            if val.ndim == 0 and val.dtype == object:
                result[key] = val.item()
            else:
                result[key] = val
        return result

def save_lc_npz(tic_id, sector, time_arr, flux_arr, flux_err, quality_arr, meta, kind='preprocessed'):
    """Save lightcurve data to a compressed npz file."""
    path = npz_path(tic_id, sector, kind=kind)
    
    # Convert meta dict to numpy object so it saves nicely
    np.savez_compressed(
        str(path),
        time=time_arr,
        flux=flux_arr,
        flux_err=flux_err,
        quality=quality_arr,
        meta=meta
    )
    return path
