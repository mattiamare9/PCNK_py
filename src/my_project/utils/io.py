"""I/O helpers
- reading Julia .jld files may require calling Julia or using h5py with care.
- Provide helper to extract jld and fallback options.
"""
import h5py, io


def try_read_jld_bytes(b: bytes):
    """Try to open bytes as HDF5. May fail on Julia-specific datatypes."""
    try:
        with h5py.File(io.BytesIO(b), 'r') as f:
            return list(f.keys())
    except Exception as e:
        return {'error': str(e)}
