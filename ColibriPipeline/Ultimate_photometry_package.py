"""
FULL PIPELINE (coords + tracking + aperture photometry + differential photometry
              + save per-star light-curve plots (GLOBAL refs)
              + ONLINE astrometry.net WCS + Gaia crossmatch
              + SNR vs Gaia G plot + SNR table
              + (NEW) IDL-style TARGET-SPECIFIC reference selection FOR EVERY STAR
                    + per-star PNG with: raw target LC + reference LC + relative LC
              + (NEW) IDL-style target-specific RELATIVE FLUX COLUMN for ALL rows
                    + SNR computed from IDL relative LC)

TRACKING MODIFICATION:
  For frame i, for each star we:
    1) cut a local box around the predicted position (from frame i-1)
    2) run DAOStarFinder in the cutout
    3) choose the detected source whose DAOStarFinder "flux" is closest to
       that star's DAOStarFinder flux in frame (i-1)
    4) update that star's stored "prev_flux" to the chosen candidate's DAO flux

Notes:
- DAOStarFinder "flux" used for tracking is NOT your aperture net_flux; it’s only for stable identity matching.
- Still recommended: set a max_match_dist_pix gate.
- For online astrometry: set env var ASTROMETRY_NET_API_KEY instead of hardcoding.

Requirements:
  pip install numpy matplotlib astropy photutils astroquery
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, join
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

from sklearn.decomposition import PCA

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from astroquery.astrometry_net import AstrometryNet
from astroquery.gaia import Gaia

import sep # SEP: Source Extraction and Photometry

import gc # for garbage collection to manage memory in long loops


# ============================
# ASTROMETRY.NET API KEY
# ============================
# IMPORTANT: do NOT hardcode real keys in shared code.
# Recommended: set env var ASTROMETRY_NET_API_KEY
ASTROMETRY_API_KEY = "lqezcrywgpzvunsn"  # leave empty; use env var


# ============================
# DEV SETTINGS
# ============================
DEV_MODE = True  # set False to ask interactively

DEV_DEFAULTS = {
    "data_directory": r"C:\Users\tonia\Downloads\NGC6031\23mai24\NGC6031\Science_Data",
    "dark_directory": r"C:\Users\tonia\Downloads\NGC6031\23mai24\Bias",

    "master_dark_fits": "master_dark.fits",
    "median_fits": "median.fits",

    # First-frame detection
    "threshold": 4,    # sigma
    "fwhm": 2.0,       # px

    # Tracking cutout
    "box_size": 15,            # odd, ideally ~6-10x FWHM
    "max_match_dist_pix": 1,   # distance gate (pix) to prevent jumps in crowded fields

    # Optional filtering to remove hot pixels / weird detections
    "use_sharpness_filter": False,
    "sharpness_min": 0.2,
    "sharpness_max": 1.0,

    "compare_jpeg": "comp.jpg",

    "coords_txt": "coords_by_frame.txt",
    "mags_txt": "mags_by_frame.txt",
    "mags_with_ref_and_diff_txt": "mags_by_frame_with_ref_and_diff.txt",

    "ref_ids_txt": "reference_star_ids.txt",
    "ref_flux_txt": "reference_flux_by_frame.txt",
    "ref_lc_flux_png": "reference_lightcurve_flux.png",
    "ref_lc_mag_png": "reference_lightcurve_mag.png",

    "aperture_r": 3,
    "sky_r_in": 8,
    "sky_r_out": 10,

    # Global reference selection (old method)
    "min_frac_present": 1.0,   # 1.0 = must exist in ALL frames
    "n_ref": None,             # None = use all stable
    "min_net_flux": 0.0,

    # Save per-star lightcurves for GLOBAL refs
    "lc_out_dir": "lightcurves",
    "lc_min_points": 5,
    "lc_dpi": 150,

    # SNR plot (NOW defaults to IDL target-specific relative)
    "snr_first_n_frames": 108,               # int, or None for "all frames"
    "snr_flux_col": "rel_flux_idl_norm",     # "net_flux" / "rel_flux" / "rel_flux_norm" / "rel_flux_idl" / "rel_flux_idl_norm"
    "snr_out_txt": "snr_vs_gaia_gmag.txt",
    "snr_out_png": "snr_vs_gaia_gmag.png",

    # Online services
    "gaia_match_radius_arcsec": 1.0,
    "gaia_extra_radius_factor": 1.3,

    # NEW: IDL-style per-target reference selection (for EVERY star)
    "idl_out_dir": "lightcurves_idlstyle",
    "idl_comp_rad_pix": 512,
    "idl_fraction_keep_pca": 0.95,
    "idl_fraction_keep_comps": 0.05,
    "idl_min_refs_per_frame": 3,
    "idl_write_ref_lists": True,

    # In the 3-panel plot:
    "idl_normalize_rel_by_median": True,    # normalize relative LC by its median (recommended)
    "idl_normalize_ref_by_median": False,   # optionally normalize ref LC too (visual aid)
}

# If False: all stars use the same denominator ref_flux (recommended)
EXCLUDE_SELF_FOR_REF_STARS = False


def get_value(prompt, default, cast=str):
    if DEV_MODE:
        print(f"{prompt}{default}  [DEV_MODE]")
        return cast(default)
    return cast(input(prompt).strip())


# ----------------------------
# I/O: FITS loading with headers
# ----------------------------

def load_fits_images_with_headers(directory):
    fits_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".fits")])
    if not fits_files:
        raise FileNotFoundError(f"No .fits files found in: {directory}")

    images, headers = [], []
    for fn in fits_files:
        path = os.path.join(directory, fn)
        with fits.open(path, memmap=False) as hdul:
            images.append(np.asarray(hdul[0].data, dtype=np.float64))
            headers.append(hdul[0].header)
    return images, headers, fits_files


def create_master_dark(dark_frames, out_path="master_dark.fits"):
    master_dark = np.median(np.stack(dark_frames, axis=0), axis=0)
    fits.PrimaryHDU(master_dark).writeto(out_path, overwrite=True)
    print(f"Master dark saved to {out_path}")
    return master_dark


def subtract_dark_frame(images, dark_frame):
    return [img - dark_frame for img in images]



# ----------------------------
# SEP background subtraction  mosaics
# ----------------------------
def sep_background_subtract_and_save_mosaics(
    images,
    out_dir="sep_background_mosaics",
    percentile=99.5,
):
    os.makedirs(out_dir, exist_ok=True)

    # We only store the results if we REALLY need them later. 
    # If this is too large, we should write them to disk immediately instead.
    subtracted = []

    for i, img in enumerate(images):
        # 1. Use float32 instead of float64. 
        # For CCD data, float32 has more than enough precision and uses HALF the RAM.
        data = np.asarray(img, dtype=np.float32)

        # 2. Perform SEP background subtraction
        bkg = sep.Background(data)
        bkg_img = bkg.back()
        data_sub = data - bkg_img

        # 3. Visualization logic
        #fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # We calculate the norm once or use a more memory-efficient way
        # By passing 'data' directly to simple_norm, it flattens it.
        #axes[0].imshow(data, norm=simple_norm(data, "sqrt", percent=percentile), cmap="Greys_r")
        #axes[0].set_title("Raw (dark-subtracted)")

        #axes[1].imshow(bkg_img, norm=simple_norm(bkg_img, "sqrt", percent=percentile), cmap="Greys_r")
        #axes[1].set_title("SEP background")

        #axes[2].imshow(data_sub, norm=simple_norm(data_sub, "sqrt", percent=percentile), cmap="Greys_r")
        #axes[2].set_title("Background-subtracted")

        #for ax in axes:
        #    ax.set_xticks([]); ax.set_yticks([])

        #plt.tight_layout()
        #plt.savefig(os.path.join(out_dir, f"frame_{i:05d}.png"), dpi=150)
        
        # 4. CRITICAL: Explicitly clear the plot and figure from memory
        #plt.clf() 
        #plt.close(fig)

        # 5. Store result as float32 to save 50% space in the list
        subtracted.append(data_sub.astype(np.float32))

        # 6. Manual cleanup every loop to free the temporary 'bkg' and 'data' objects
        del data_sub  # Delete the large array once you are done with it
        del data
        del bkg
        del bkg_img
        gc.collect() 

    print(f"SEP background mosaics saved to: {out_dir}")
    return subtracted


def create_median_image(images, output_file):
    median_image = np.median(np.stack(images, axis=0), axis=0)
    fits.PrimaryHDU(median_image).writeto(output_file, overwrite=True)
    return median_image


def process_photometry_data(data_directory, dark_directory, median_output, master_dark_path):
    dark_frames, _, _ = load_fits_images_with_headers(dark_directory)
    master_dark = create_master_dark(dark_frames, out_path=master_dark_path)

    data_images, data_headers, data_files = load_fits_images_with_headers(data_directory)
    print(f"Loaded {len(data_images)} data frames.")

    subtracted_images = subtract_dark_frame(data_images, master_dark)

    # --- NEW: SEP background subtraction per frame + mosaics
    subtracted_images = sep_background_subtract_and_save_mosaics(
        subtracted_images,
        out_dir="sep_background_mosaics",
    )

    median_img = create_median_image(subtracted_images, median_output)
    print(f"Median image saved to {median_output}")

    return subtracted_images, data_headers, median_img, data_files


# ----------------------------
# Plot helpers
# ----------------------------

def _plot_circles(ax, positions_xy, r=10.0, lw=1.2, alpha=0.6, max_stars=None):
    pos = np.asarray(positions_xy, dtype=float)
    ok = np.isfinite(pos).all(axis=1)
    pos = pos[ok]
    if max_stars is not None and len(pos) > max_stars:
        pos = pos[:max_stars]

    for x, y in pos:
        ax.add_patch(plt.Circle((x, y), r, fill=False, color="red", lw=lw, alpha=alpha))


def save_first_last_side_by_side(first_img, last_img, pos_first, pos_last, out_jpeg,
                                 r=10.0, max_stars=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(first_img, norm=simple_norm(first_img, "sqrt", percent=99.8), cmap="Greys_r")
    _plot_circles(axes[0], pos_first, r=r, max_stars=max_stars)
    axes[0].set_title("First frame (original coords)")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")

    axes[1].imshow(last_img, norm=simple_norm(last_img, "sqrt", percent=99.8), cmap="Greys_r")
    _plot_circles(axes[1], pos_last, r=r, max_stars=max_stars)
    axes[1].set_title("Last frame (refined coords)")
    axes[1].set_xlabel("x [pix]")
    axes[1].set_ylabel("y [pix]")

    plt.tight_layout()
    plt.savefig(out_jpeg, dpi=200)
    plt.close(fig)
    print(f"Saved side-by-side plot to {out_jpeg}")

def save_tracking_debug_images(subtracted_images, positions_per_frame, out_dir="tracking_mosaics", r=3.0, dpi=150):
    """
    Saves each background-subtracted frame with circled detections.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving tracking debug images to: {out_dir}")

    for i, (img, pos) in enumerate(zip(subtracted_images, positions_per_frame)):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use simple_norm for consistent scaling
        norm = simple_norm(img, "sqrt", percent=99.5)
        ax.imshow(img, norm=norm, cmap="Greys_r")
        
        # Reuse your existing circle plotting helper
        _plot_circles(ax, pos, r=r)
        
        ax.set_title(f"Frame {i:05d} - Tracked Stars")
        ax.set_axis_off()
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, f"track_frame_{i:05d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)
        
        if (i + 1) % 50 == 0:
            print(f"  ...saved {i+1}/{len(subtracted_images)} tracking images")


# ----------------------------
# SAVE PER-STAR LIGHTCURVES (GLOBAL refs output)
# ----------------------------

def save_lightcurves_for_all_stars_flux_only(
    mags_table,
    out_dir="lightcurves",
    id_col="id",
    jd_col="JD",
    rel_flux_col="rel_flux",
    raw_flux_col="net_flux",
    min_points=5,
    dpi=150,
):
    os.makedirs(out_dir, exist_ok=True)

    ids = np.unique(np.array(mags_table[id_col], dtype=int))
    ids = np.sort(ids)

    n_saved = 0
    for sid in ids:
        rows = mags_table[np.array(mags_table[id_col], dtype=int) == int(sid)]
        if len(rows) == 0:
            continue

        jd = np.array(rows[jd_col], dtype=float)
        s = np.argsort(jd)
        jd = jd[s]

        rel_flux = np.array(rows[rel_flux_col], dtype=float)[s] if rel_flux_col in rows.colnames else np.full(len(rows), np.nan)
        raw_flux = np.array(rows[raw_flux_col], dtype=float)[s] if raw_flux_col in rows.colnames else np.full(len(rows), np.nan)

        ok_rel = np.isfinite(rel_flux)
        ok_raw = np.isfinite(raw_flux)

        if ok_rel.sum() < min_points and ok_raw.sum() < min_points:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        axes[0].plot(jd, rel_flux, marker="o", linestyle="-", markersize=3)
        axes[0].set_ylabel(rel_flux_col)
        axes[0].set_title(f"Star id={sid}")

        axes[1].plot(jd, raw_flux, marker="o", linestyle="-", markersize=3)
        axes[1].set_ylabel(raw_flux_col)
        axes[1].set_xlabel("JD")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"star_{sid:05d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)
        n_saved += 1

    print(f"Saved {n_saved} light-curve plots to: {out_dir}")


# ----------------------------
# DAOStarFinder: detection on first frame
# ----------------------------

def find_stars_first_frame_with_flux_daofind(image, threshold_sigma=5.0, fwhm=2.5, margin=10):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(image - median)

    if sources is None or len(sources) == 0:
        raise RuntimeError("DAOStarFinder found no sources on the first frame.")

    x0 = np.asarray(sources["xcentroid"], float)
    y0 = np.asarray(sources["ycentroid"], float)

    h, w = image.shape

    # keep stars far enough from the edges
    mask = (
        (x0 > margin) &
        (x0 < w - margin) &
        (y0 > margin) &
        (y0 < h - margin)
    )

    x0 = x0[mask]
    y0 = y0[mask]
    pos = np.column_stack([x0, y0])

    # flux handling
    if "flux" in sources.colnames:
        flux0 = np.asarray(sources["flux"], float)[mask]
    elif "peak" in sources.colnames:
        flux0 = np.asarray(sources["peak"], float)[mask]
    else:
        flux0 = np.full(len(pos), np.nan, dtype=float)

    print(f"Detected {len(pos)} stars after edge filtering.")
    return pos, flux0

def find_stars_first_frame_with_flux(image, threshold_sigma=5.0, fwhm=2.5, margin=10):
    print("  [SEP] Detecting master stars on first frame...")
    bkg = sep.Background(image)
    data_sub = image - bkg
    thresh = threshold_sigma * bkg.globalrms
    
    # Extract objects and the segmentation map
    objects, seg = sep.extract(data_sub, thresh, minarea=3, segmentation_map=True)
    # Convert all star IDs (>0) to 1, keep background at 0
    binary_seg = (seg > 0).astype(np.uint8)

    # Save as Black (background) and White (stars)
    plt.imsave("binary_segmentation_map.png", binary_seg, cmap='gray')

    
    x0, y0 = objects['x'], objects['y']
    flux0 = objects['flux']   # Integrated brightness
    peak0 = objects['peak']   # Brightest pixel (for saturation check)
    axis_ratio_all = objects['b'] / objects['a']

    h, w = image.shape
    mask = (x0 > margin) & (x0 < w - margin) & (y0 > margin) & (y0 < h - margin)

    # Filter and synchronize all metadata
    pos = np.column_stack([x0[mask], y0[mask]])
    flux_filt = flux0[mask]
    ar_filt = axis_ratio_all[mask]
    peak_filt = peak0[mask]

    print(f"  [SEP] Master List created: {len(pos)} stars detected.")
    # Return everything needed to build the guide mask
    return pos, flux_filt, ar_filt, peak_filt

# ----------------------------
# TRACKING: choose candidate with closest flux to previous frame
# ----------------------------

def refine_positions_daofind_local_fluxmatch(
    image_next,
    positions_xy_guess,
    prev_flux,
    is_guide,  # Pre-calculated boolean mask
    fwhm=3.0,
    threshold_sigma=5.0,
    box_size=31,
    max_match_dist_pix=6.0,
    frame_idx=0
):
    h, w = image_next.shape
    half = box_size // 2
    if image_next.dtype.byteorder not in ('=', '|'):
        image_next = image_next.byteswap().newbyteorder()
    
    # Identify indices of our guides
    guide_indices = np.where(is_guide)[0]
    guide_dx, guide_dy = [], []
    guide_x_active, guide_y_active = [], []

    seg_dir = f"segments_frame_{frame_idx:03d}"
    if not os.path.exists(seg_dir): os.makedirs(seg_dir)

    # --- PASS 1: MEASURE SHIFTS FROM MASTER GUIDES ---
    for k in guide_indices:
        xg, yg = positions_xy_guess[k]
        if not (np.isfinite(xg) and np.isfinite(yg)): continue
        
        x1, x2 = max(0, int(xg - half)), min(w, int(xg + half + 1))
        y1, y2 = max(0, int(yg - half)), min(h, int(yg + half + 1))
        cut = np.ascontiguousarray(image_next[y1:y2, x1:x2], dtype=np.float32)
        
        try:
            mask_path = f"{seg_dir}/star_{k:05d}_mask.png"
            bkg = sep.Background(cut)
            res = sep.extract(cut, threshold_sigma * bkg.globalrms, minarea=3, segmentation_map=True)
            objs, seg = res[0], res[1]
            plt.imsave(mask_path, (seg > 0).astype(np.uint8), cmap='gray')

            if objs is not None and len(objs) > 0:
                cx_local, cy_local = xg - x1, yg - y1
                dist = np.sqrt((objs['x'] - cx_local)**2 + (objs['y'] - cy_local)**2)
                idx = np.argmin(dist)
                
                if dist[idx] <= max_match_dist_pix:
                    guide_dx.append(objs['x'][idx] - cx_local)
                    guide_dy.append(objs['y'][idx] - cy_local)
                    guide_x_active.append(xg)
                    guide_y_active.append(yg)

                    # Solid Black Background / Solid White Star
                    binary_mask = (seg > 0).astype(np.uint8)
                    plt.imsave(f"{seg_dir}/star_{k:05d}_mask.png", binary_mask, cmap='gray', vmin=0, vmax=1)
        except: continue

    print(f"[Frame {frame_idx}] guides active: {len(guide_dx)} / {np.sum(is_guide)}")

    # --- 2. CALCULATE GLOBAL MEDIAN SHIFT ---
    if len(guide_dx) > 0:
        avg_dx, avg_dy = np.median(guide_dx), np.median(guide_dy)
        mags = np.sqrt(np.array(guide_dx)**2 + np.array(guide_dy)**2)
        
        plt.figure(figsize=(10, 8))
        # Master constellation (Grey)
        plt.scatter(positions_xy_guess[is_guide, 0], positions_xy_guess[is_guide, 1], 
                    c='lightgrey', s=5, alpha=0.3)
        # Active Successful Guides (Color)
        sc = plt.scatter(guide_x_active, guide_y_active, c=mags, cmap='magma', 
                         s=15, vmin=0, vmax=2.0)
        plt.colorbar(sc, label='Shift (pix)')
        plt.xlim(0, w); plt.ylim(0, h)
        plt.title(f"Frame {frame_idx} Shift Map\nMedian: dx={avg_dx:.3f}, dy={avg_dy:.3f}")
        plt.savefig(f"global_shift_map_frame_{frame_idx:03d}.png")
        plt.close()
    else:
        avg_dx, avg_dy = 0.0, 0.0

    # --- 3. APPLY RIGID SHIFT TO EVERYONE ---
    refined = positions_xy_guess + np.array([avg_dx, avg_dy])
    success = np.isfinite(positions_xy_guess).all(axis=1)
    
    return refined, success, np.array(prev_flux)

def track_positions_rigid_registration(images, master_positions, upsample_factor=50):
    """
    Rigid image registration tracker.
    
    - Frame 0 is the reference (no shift)
    - Each frame is registered to frame 0
    - Master catalog is never modified
    - Only a global (dx, dy) is applied
    
    Returns:
        positions_per_frame : list of arrays (Nstars, 2)
        shifts              : list of (dx, dy)
    """

    ref = np.asarray(images[0], dtype=np.float32)
    master_positions = np.asarray(master_positions, dtype=np.float64)

    positions_per_frame = [master_positions.copy()]
    shifts = [(0.0, 0.0)]

    print("\nRigid registration tracking:")
    print("Frame 0 shift = (0.000, 0.000)")

    for i in range(1, len(images)):
        img = np.asarray(images[i], dtype=np.float32)

        shift, error, _ = phase_cross_correlation(
            ref,
            img,
            upsample_factor=upsample_factor
        )

        # phase_cross_correlation returns (dy, dx)
        dy, dx = shift
        dx = -dx
        dy = -dy

        shifted_positions = master_positions + np.array([dx, dy])

        positions_per_frame.append(shifted_positions)
        shifts.append((dx, dy))

        print(f"Frame {i:04d} shift = ({dx:+.4f}, {dy:+.4f})")

    return positions_per_frame, shifts


def track_positions_refine_fluxmatch(
    images, positions0, flux0, axis_ratio, peak0,
    ellip_max=0.2, 
    guide_flux_min=500,
    guide_peak_max=20000,
    **kwargs
):
    # 1. CREATE THE MASTER GUIDE MASK (Only once)
    ellipticity = np.abs(1.0 - np.nan_to_num(axis_ratio, nan=0.0))
    is_guide = (ellipticity < ellip_max) & (flux0 > guide_flux_min) & (peak0 < guide_peak_max)
    
    # Store the absolute master positions from frame 0
    master_positions = np.asarray(positions0, dtype=np.float64)
    positions_per_frame = [master_positions.copy()]
    
    # Track the cumulative shift from the first frame

    for i in range(len(images) - 1):
        # Calculate shift for frame i+1 relative to the PREVIOUS frame's positions
        # but using the fixed master guide list
        next_pos, success, _ = refine_positions_daofind_local_fluxmatch(
            images[i + 1],
            positions_per_frame[-1], # Look where we last were
            flux0,
            is_guide=is_guide,
            frame_idx=i + 1,
            **kwargs
        )
        
        # We don't want 'next_pos' to be a mix of local detections.
        # We want the WHOLE CATALOG moved by the median shift found by the guides.
        positions_per_frame.append(next_pos)
        
    return positions_per_frame



# ----------------------------
# Time + photometry
# ----------------------------

def get_time_from_header(hdr):
    date_obs = hdr.get("DATE-OBS")
    jd = hdr.get("JD")

    # Try JD directly
    if jd is not None:
        try:
            return str(date_obs), float(jd)
        except Exception:
            pass

    # Fix invalid DATE-OBS like 2016-00-01
    if date_obs:
        fixed = date_obs.replace("-00-", "-01-")
        try:
            t = Time(fixed, format="isot", scale="utc")
            return str(date_obs), t.jd
        except Exception:
            pass

    return str(date_obs), np.nan


def aperture_photometry_sequence(images, headers, positions_per_frame,
                                 aperture_r=3.0, sky_r_in=6.0, sky_r_out=9.0,
                                 coords_out="coords_by_frame.txt",
                                 mags_out="mags_by_frame.txt"):
    if not (len(images) == len(headers) == len(positions_per_frame)):
        raise ValueError("images, headers, positions_per_frame must have the same length")

    coord_rows = []
    mag_rows = []

    for frame_idx, (img, hdr, pos_xy_all) in enumerate(zip(images, headers, positions_per_frame)):
        date_obs, jd = get_time_from_header(hdr)

        pos_xy_all = np.asarray(pos_xy_all, dtype=float)
        ok = np.isfinite(pos_xy_all).all(axis=1)
        pos_xy = pos_xy_all[ok]
        ids = np.nonzero(ok)[0].astype(int)

        if len(pos_xy) == 0:
            continue

        for sid, (x, y) in zip(ids, pos_xy):
            coord_rows.append((frame_idx, int(sid), float(x), float(y), date_obs, float(jd)))

        aper = CircularAperture(pos_xy, r=aperture_r)
        ann = CircularAnnulus(pos_xy, r_in=sky_r_in, r_out=sky_r_out)

        phot_aper = aperture_photometry(img, aper, method="exact")
        phot_ann = aperture_photometry(img, ann, method="exact")

        aper_sum = np.asarray(phot_aper["aperture_sum"], dtype=float)
        ann_sum = np.asarray(phot_ann["aperture_sum"], dtype=float)

        aper_area = float(aper.area)
        ann_area = float(ann.area)

        sky_per_pix = ann_sum / ann_area
        sky_in_aper = sky_per_pix * aper_area
        net_flux = aper_sum - sky_in_aper

        mag_inst = np.full_like(net_flux, np.nan, dtype=float)
        good = net_flux > 0
        mag_inst[good] = -2.5 * np.log10(net_flux[good])

        for sid, m, nf in zip(ids, mag_inst, net_flux):
            mag_rows.append((frame_idx, int(sid),
                             float(m) if np.isfinite(m) else np.nan,
                             float(nf), date_obs, float(jd)))

    coords_tab = Table(rows=coord_rows, names=["frame", "id", "x", "y", "DATE_OBS", "JD"])
    coords_tab.write(coords_out, format="ascii.tab", overwrite=True)
    print(f"Wrote coordinates table -> {coords_out}")

    mags_tab = Table(rows=mag_rows, names=["frame", "id", "mag_inst", "net_flux", "DATE_OBS", "JD"])
    mags_tab.write(mags_out, format="ascii.tab", overwrite=True)
    print(f"Wrote magnitudes table -> {mags_out}")

    return coords_tab, mags_tab


# ----------------------------
# Global reference ensemble + differential (old method)
# ----------------------------

def select_reference_star_ids(
    mags_tab,
    n_ref=None,
    min_frac_present=1.0,
    min_net_flux=0.0,
    keep_percentile_range=(25, 75),
):
    frames = np.unique(np.array(mags_tab["frame"]))
    n_frames = len(frames)
    ids = np.unique(np.array(mags_tab["id"]))

    presence_frac = {}
    median_flux = {}

    for sid in ids:
        rows = mags_tab[mags_tab["id"] == sid]
        nf = np.array(rows["net_flux"], dtype=float)
        ok = np.isfinite(nf) & (nf > min_net_flux)

        present_count = int(np.sum(ok))
        presence_frac[sid] = present_count / n_frames if n_frames > 0 else 0.0
        median_flux[sid] = float(np.nanmedian(nf[ok])) if present_count > 0 else -np.inf

    stable_ids = [sid for sid in ids if presence_frac[sid] >= min_frac_present and median_flux[sid] > 0]

    if keep_percentile_range is not None and len(stable_ids) > 0:
        p_lo, p_hi = keep_percentile_range
        vals = np.array([median_flux[sid] for sid in stable_ids], dtype=float)
        lo = np.nanpercentile(vals, p_lo)
        hi = np.nanpercentile(vals, p_hi)
        stable_ids = [sid for sid in stable_ids if (median_flux[sid] >= lo and median_flux[sid] <= hi)]

    if n_ref is not None and len(stable_ids) > n_ref:
        stable_ids = sorted(stable_ids, key=lambda s: median_flux[s], reverse=True)[:n_ref]

    return np.array(stable_ids, dtype=int)


def write_reference_star_list(ref_ids, out_path="reference_star_ids.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Reference star IDs\n")
        for sid in ref_ids:
            f.write(f"{int(sid)}\n")
    print(f"Wrote reference star list -> {out_path}")


def compute_reference_flux_table(mags_tab, ref_ids, min_net_flux=0.0):
    frames = np.unique(np.array(mags_tab["frame"]))
    ref_ids = np.array(ref_ids, dtype=int)

    out_rows = []
    for fr in frames:
        rows_f = mags_tab[mags_tab["frame"] == fr]
        if len(rows_f) == 0:
            continue

        id_f = np.array(rows_f["id"], dtype=int)
        nf_f = np.array(rows_f["net_flux"], dtype=float)
        flux_map = {i: nf for i, nf in zip(id_f, nf_f)}

        ref_fluxes = np.array([flux_map.get(int(i), np.nan) for i in ref_ids], dtype=float)
        ok = np.isfinite(ref_fluxes) & (ref_fluxes > min_net_flux)
        n_used = int(np.sum(ok))

        ref_sum = float(np.nansum(ref_fluxes[ok])) if n_used > 0 else np.nan
        ref_mean = float(np.nanmean(ref_fluxes[ok])) if n_used > 0 else np.nan

        ref_mag_sum = float(-2.5 * np.log10(ref_sum)) if np.isfinite(ref_sum) and ref_sum > 0 else np.nan
        ref_mag_mean = float(-2.5 * np.log10(ref_mean)) if np.isfinite(ref_mean) and ref_mean > 0 else np.nan

        date_obs = str(rows_f["DATE_OBS"][0])
        jd = float(rows_f["JD"][0])

        out_rows.append((int(fr), date_obs, jd,
                         ref_sum, ref_mean, n_used,
                         ref_mag_sum, ref_mag_mean))

    return Table(
        rows=out_rows,
        names=["frame", "DATE_OBS", "JD",
               "ref_flux_sum", "ref_flux_mean", "n_ref_used",
               "ref_mag_sum", "ref_mag_mean"]
    )


def add_differential_columns(mags_tab, ref_flux_tab, ref_ids,
                             denom_mode="mean",
                             exclude_self_for_refstars=False):
    ref_set = set(int(x) for x in np.array(ref_ids, dtype=int))

    denom_col = "ref_flux_mean" if denom_mode == "mean" else "ref_flux_sum"
    frame_to_denom = {int(r["frame"]): float(r[denom_col]) for r in ref_flux_tab}

    ids = np.array(mags_tab["id"], dtype=int)
    frames = np.array(mags_tab["frame"], dtype=int)
    net_flux = np.array(mags_tab["net_flux"], dtype=float)

    is_ref = np.array([int(i) in ref_set for i in ids], dtype=bool)

    denom = np.array([frame_to_denom.get(int(fr), np.nan) for fr in frames], dtype=float)

    if exclude_self_for_refstars and denom_mode == "sum":
        idx = np.where(is_ref)[0]
        denom[idx] = denom[idx] - net_flux[idx]

    denom[~np.isfinite(denom) | (denom <= 0)] = np.nan

    rel_flux = np.full(len(net_flux), np.nan, dtype=float)
    good = np.isfinite(net_flux) & (net_flux > 0) & np.isfinite(denom) & (denom > 0)
    rel_flux[good] = net_flux[good] / denom[good]

    diff_mag = np.full(len(rel_flux), np.nan, dtype=float)
    good2 = np.isfinite(rel_flux) & (rel_flux > 0)
    diff_mag[good2] = -2.5 * np.log10(rel_flux[good2])

    mags2 = mags_tab.copy()
    mags2["is_ref"] = is_ref
    mags2["ref_denom"] = denom
    mags2["rel_flux"] = rel_flux
    mags2["diff_mag"] = diff_mag
    return mags2


def normalize_rel_flux_per_star(tab, id_col="id", rel_flux_col="rel_flux",
                                out_col="rel_flux_norm"):
    ids = np.array(tab[id_col], dtype=int)
    rf = np.array(tab[rel_flux_col], dtype=float)

    out = np.full(len(tab), np.nan, dtype=float)

    for sid in np.unique(ids):
        m = (ids == sid) & np.isfinite(rf) & (rf > 0)
        if np.sum(m) < 3:
            continue
        med = np.median(rf[m])
        if np.isfinite(med) and med > 0:
            out[m] = rf[m] / med

    tab2 = tab.copy()
    tab2[out_col] = out
    return tab2


def plot_reference_lightcurve(ref_flux_tab, out_flux_png, out_mag_png, use="mean"):
    jd = np.array(ref_flux_tab["JD"], dtype=float)

    if use.lower() == "sum":
        flux = np.array(ref_flux_tab["ref_flux_sum"], dtype=float)
        mag = np.array(ref_flux_tab["ref_mag_sum"], dtype=float)
        title_suffix = " (sum)"
    else:
        flux = np.array(ref_flux_tab["ref_flux_mean"], dtype=float)
        mag = np.array(ref_flux_tab["ref_mag_mean"], dtype=float)
        title_suffix = " (mean)"

    med = np.nanmedian(flux)
    flux_norm = flux / med if np.isfinite(med) and med != 0 else flux

    plt.figure(figsize=(10, 5))
    plt.plot(jd, flux_norm, marker="o", linestyle="-")
    plt.xlabel("JD")
    plt.ylabel("Reference flux (normalized)")
    plt.title("Reference ensemble light curve" + title_suffix)
    plt.tight_layout()
    plt.savefig(out_flux_png, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(jd, mag, marker="o", linestyle="-")
    plt.gca().invert_yaxis()
    plt.xlabel("JD")
    plt.ylabel("Reference mag")
    plt.title("Reference ensemble magnitude" + title_suffix)
    plt.tight_layout()
    plt.savefig(out_mag_png, dpi=200)
    plt.close()

    print(f"Saved reference light curves -> {out_flux_png} and {out_mag_png}")


# ----------------------------
# IDL-style per-target selection + helpers
# ----------------------------

def build_flux_matrix(mags_tab, flux_col="net_flux"):
    ids_sorted = np.sort(np.unique(np.array(mags_tab["id"], dtype=int)))
    frames_sorted = np.sort(np.unique(np.array(mags_tab["frame"], dtype=int)))

    id_to_i = {int(sid): i for i, sid in enumerate(ids_sorted)}
    fr_to_t = {int(fr): t for t, fr in enumerate(frames_sorted)}

    # one JD per frame
    jd_per_frame = np.full(len(frames_sorted), np.nan, dtype=float)
    for row in mags_tab:
        fr = int(row["frame"])
        jd = float(row["JD"])
        t = fr_to_t[fr]
        if not np.isfinite(jd_per_frame[t]) and np.isfinite(jd):
            jd_per_frame[t] = jd

    F = np.full((len(ids_sorted), len(frames_sorted)), np.nan, dtype=float)
    for row in mags_tab:
        sid = int(row["id"])
        fr = int(row["frame"])
        f = float(row[flux_col])
        if np.isfinite(f):
            F[id_to_i[sid], fr_to_t[fr]] = f

    return ids_sorted, frames_sorted, jd_per_frame, F

def sigma_clip_lightcurves(Fn, sigma=5.0, iters=2):
    """
    Sigma-clip outlier time points in normalized light curves.

    Fn : (Nstars, Tframes)
    Returns cleaned array with NaNs where outliers were.
    """
    Fn_clean = Fn.copy()

    for _ in range(iters):
        med = np.nanmedian(Fn_clean, axis=1, keepdims=True)
        resid = Fn_clean - med
        mad = np.nanmedian(np.abs(resid), axis=1, keepdims=True)

        # robust sigma estimate
        robust_sigma = 1.4826 * mad
        mask = np.abs(resid) > sigma * robust_sigma

        Fn_clean[mask] = np.nan

    return Fn_clean


def select_refs_idl_relative_for_target(
    target_i,
    F,
    positions0,
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_keep=3,
    return_stats=False,
):
    N, T = F.shape

    valid = np.isfinite(F) & (F > float(min_net_flux))
    present_frac = valid.sum(axis=1) / max(T, 1)

    cand = present_frac >= float(min_frac_present)

    # distance cut around target
    if positions0 is not None:
        positions0 = np.asarray(positions0, float)
        tx, ty = positions0[target_i]
        d = np.sqrt((positions0[:, 0] - tx) ** 2 + (positions0[:, 1] - ty) ** 2)
        cand &= (d <= float(comp_rad_pix))

    cand[target_i] = False
    cand_idx = np.where(cand)[0]

    num_candidates = len(cand_idx)

    if len(cand_idx) < min_keep:
        if return_stats:
            return np.array([], dtype=int), {
                "total_in_radius": len(cand_idx),
                "selected": 0,
                "rejected": len(cand_idx)
            }
        return np.array([], dtype=int)

    means = np.nanmean(np.where(valid, F, np.nan), axis=1)
    Fn = F / means[:, None]

    # PASS 1: compare candidates to group mean curve
    Fn_c = Fn[cand_idx, :]
    ref_curve1 = np.nanmean(Fn_c, axis=0)
    metric1 = np.nansum((Fn_c - ref_curve1[None, :]) ** 2, axis=1)
    order1 = np.argsort(metric1)

    n_keep1 = int(round(float(fraction_keep_pca) * len(cand_idx)))
    n_keep1 = max(min_keep, min(n_keep1, len(cand_idx)))
    keep1_idx = cand_idx[order1[:n_keep1]]

    if len(keep1_idx) < min_keep:
        if return_stats:
            return np.array([], dtype=int), {
                "total_in_radius": len(cand_idx),
                "selected": 0,
                "rejected": len(cand_idx)
            }
        return np.array([], dtype=int)
    

    # ... (Previous code remains the same up to keep1_idx)

    # --- NEW PCA STEP ---
    # Fn_k contains the normalized light curves of the first-pass survivors
                    # target_curve = Fn[target_i, :].reshape(1, -1)
                    # target_clean = sigma_clip_lightcurves(target_curve, sigma=5.0, iters=2)
                    # target_input = np.where(np.isfinite(target_clean), target_clean, 1.0)
                    
                    # # --- PCA preparation ---
                    # Fn_k = Fn[keep1_idx, :]
                    # target_curve = Fn[target_i, :].reshape(1, -1)

                    # Fn_subset = np.vstack([target_curve, Fn_k])
                    # Fn_subset_clean, _ = pca_remove_common_modes_no_sigma(
                    #     Fn_subset,
                    #     max_pcomps=3,
                    #     sigma_clip_sigma=2.0,
                    #     sigma_clip_iters=2,
                    #     center_value=1.0,
                    # )

                    # target_clean = Fn_subset_clean[0, :]
                    # Fn_k_clean = Fn_subset_clean[1:, :]

                    # metric2 = np.nansum((Fn_k_clean - target_clean[None, :]) ** 2, axis=1)
                    # order2 = np.argsort(metric2)

    # clean candidate curves
    #Fn_k_clean = sigma_clip_lightcurves(Fn_k, sigma=2.0, iters=2)
    #Fn_pca_input = np.where(np.isfinite(Fn_k_clean), Fn_k_clean, 1.0)

    # clean target curve
    #target_curve = Fn[target_i, :].reshape(1, -1)
    #target_clean = sigma_clip_lightcurves(target_curve, sigma=5.0, iters=2)
    #target_input = np.where(np.isfinite(target_clean), target_clean, 1.0)

    # dynamic PCA size (prevents crash)
    #n_components = min(3, min(Fn_pca_input.shape))
    #pca = PCA(n_components=n_components)
    #pca.fit(Fn_pca_input)

    #Fn_k_projected = pca.inverse_transform(pca.transform(Fn_pca_input))
    #target_projected = pca.inverse_transform(pca.transform(target_input)).flatten()

    #Fn_k_detrended = Fn_pca_input - Fn_k_projected + 1.0
    #target_detrended = target_input - target_projected + 1.0

    #metric2 = np.nansum((Fn_k_detrended - target_detrended[None, :]) ** 2, axis=1)
    #order2 = np.argsort(metric2)

    # ... (Continue with n_keep2 logic)

    # PASS 2: compare to TARGET curve
    target_curve = Fn[target_i, :]
    Fn_k = Fn[keep1_idx, :]
    metric2 = np.nansum((Fn_k - target_curve[None, :]) ** 2, axis=1)
    order2 = np.argsort(metric2)

    n_keep2 = int(round(float(fraction_keep_comps) * len(keep1_idx)))
    n_keep2 = max(min_keep, min(n_keep2, len(keep1_idx)))
    ref_idx = keep1_idx[order2[:n_keep2]]

    # --- Return handling ---
    if return_stats:
        stats = {
            "total_in_radius": num_candidates,
            "selected": len(ref_idx),
            "rejected": num_candidates - len(ref_idx)
        }
        return ref_idx, stats

    return np.array(ref_idx, dtype=int)


def compute_rel_curve_pca_postselect(
    F,
    target_i,
    ref_idx,
    min_net_flux=0.0,
    min_refs_per_frame=3,
    max_pcomps=2,
    return_pca_features=False,
):
    """
    PCA happens AFTER final ref_idx selection.
    IMPORTANT: Fit PCA on REFS ONLY (prevents self-subtraction), then apply the
    same PCA model to detrend BOTH refs and target, then divide.

    Returns:
      rel (T,), ref_mean (T,), Fn_t_clean (T,), Fn_r_clean (R,T)
      plus (optional): components_TK (T,K), scores_MK (R,K), target_scores (K,)
    """
    import numpy as np
    from sklearn.decomposition import PCA

    F = np.asarray(F, float)
    N, T = F.shape

    valid = np.isfinite(F) & (F > float(min_net_flux))
    means = np.nanmean(np.where(valid, F, np.nan), axis=1)
    means = np.where(np.isfinite(means) & (means > 0), means, np.nan)

    Fn = F / means[:, None]  # normalized ~1

    Fn_t = Fn[target_i, :]
    Fn_r = Fn[np.asarray(ref_idx, int), :]  # (R,T)

    # mild outlier clipping (optional) — I'd keep refs clip, disable target clip if you have real variables
    try:
        Fn_r = sigma_clip_lightcurves(Fn_r, sigma=4.0, iters=1)
        # safer for variables/events:
        # Fn_t = sigma_clip_lightcurves(Fn_t.reshape(1, -1), sigma=20.0, iters=1).reshape(-1)
    except Exception:
        pass

    # ---------- Fit PCA on refs ONLY ----------
    Fn_subset = Fn_r  # (R,T)

    # impute NaNs with per-star median
    med_r = np.nanmedian(Fn_subset, axis=1, keepdims=True)
    med_r = np.where(np.isfinite(med_r), med_r, 1.0)
    Fn_r_imp = np.where(np.isfinite(Fn_subset), Fn_subset, med_r)

    Xr = Fn_r_imp - 1.0  # center (R,T)

    R = Xr.shape[0]
    K = int(min(max_pcomps, R, T))

    if K >= 1:
        pca = PCA(n_components=K)
        pca.fit(Xr)

        # Refs: reconstruct common-mode and subtract
        Xr_common = pca.inverse_transform(pca.transform(Xr))
        Xr_clean = Xr - Xr_common
        Fn_r_clean = Xr_clean + 1.0

        # Target: project onto the SAME PCA basis and subtract
        Fn_t_imp = np.where(np.isfinite(Fn_t), Fn_t, np.nanmedian(Fn_t))
        Xt = (Fn_t_imp - 1.0).reshape(1, -1)  # (1,T)
        Xt_common = pca.inverse_transform(pca.transform(Xt)).reshape(-1)
        Fn_t_clean = (Xt.reshape(-1) - Xt_common) + 1.0

        # restore NaNs where original invalid (optional, but keeps masking behavior)
        Fn_r_clean = np.where(np.isfinite(Fn_subset), Fn_r_clean, np.nan)
        Fn_t_clean = np.where(np.isfinite(Fn_t), Fn_t_clean, np.nan)

        # PCA feature outputs
        components_TK = pca.components_.T       # (T,K) time-series PCs
        scores_MK = pca.transform(Xr)           # (R,K) ref coefficients
        target_scores = pca.transform(Xt).reshape(-1)  # (K,)
    else:
        Fn_t_clean = Fn_t.copy()
        Fn_r_clean = Fn_r.copy()
        components_TK = None
        scores_MK = None
        target_scores = None

    # ---------- Build reference mean and relative curve ----------
    ok_ref = np.isfinite(Fn_r_clean) & (Fn_r_clean > 0)
    n_ok = ok_ref.sum(axis=0)

    ref_mean = np.full(T, np.nan, dtype=float)
    good_frames = n_ok >= int(min_refs_per_frame)
    if np.any(good_frames):
        ref_mean[good_frames] = np.nanmean(np.where(ok_ref, Fn_r_clean, np.nan), axis=0)[good_frames]

    rel = np.full(T, np.nan, dtype=float)
    good = np.isfinite(Fn_t_clean) & np.isfinite(ref_mean) & (ref_mean > 0)
    rel[good] = Fn_t_clean[good] / ref_mean[good]

    if return_pca_features:
        return rel, ref_mean, Fn_t_clean, Fn_r_clean, components_TK, scores_MK, target_scores

    return rel, ref_mean, Fn_t_clean, Fn_r_clean

def save_pca_feature_plot(out_path, jd, components_TK, scores_MK, ids_M=None, title="PCA features"):
    """
    components_TK: (T, K) time-series PCs (same length as jd)
    scores_MK:     (M, K) per-star coefficients (rows: refs only, in your current call)
    ids_M:         list/array length M of star IDs (optional)
    """

    jd = np.asarray(jd, float)
    components_TK = np.asarray(components_TK, float)
    scores_MK = np.asarray(scores_MK, float)

    T, K = components_TK.shape
    M = scores_MK.shape[0]

    if jd.shape[0] != T:
        raise ValueError(f"jd length ({jd.shape[0]}) must match components_TK rows ({T}).")

    # --- Make K PC axes that share x with each other ---
    fig = plt.figure(figsize=(10, 2.2 * (K + 1)))
    gs = fig.add_gridspec(nrows=K + 1, ncols=1, height_ratios=[1] * K + [1.2])

    pc_axes = []
    first_ax = None
    for k in range(K):
        ax = fig.add_subplot(gs[k, 0], sharex=first_ax) if first_ax is not None else fig.add_subplot(gs[k, 0])
        if first_ax is None:
            first_ax = ax
        pc_axes.append(ax)

        ax.plot(jd, components_TK[:, k], marker="o", linestyle="-", markersize=2)
        ax.set_ylabel(f"PC{k+1}(t)")
        ax.grid(True, alpha=0.2)

    # --- Bar axis does NOT share x (critical fix) ---
    axb = fig.add_subplot(gs[K, 0])
    x = np.arange(M)
    axb.bar(x, scores_MK[:, 0])

    if ids_M is not None:
        labels = [str(int(v)) for v in ids_M]
    else:
        labels = [f"ref{i}" for i in range(M)]

    axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=90, fontsize=7)
    axb.set_ylabel("coeff (PC1)")
    axb.set_title("Per-star PC1 coefficients (selected refs)")
    axb.grid(True, axis="y", alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def pca_remove_common_modes_no_sigma(
    Fn_subset,
    max_pcomps=2,
    sigma_clip_sigma=5.0,
    sigma_clip_iters=2,
    center_value=1.0,
):
    """
    Remove common trends using PCA, WITHOUT per-star sigma/error normalization.

    Fn_subset: (M, T) normalized curves (mean ~ 1). Can contain NaNs.
    Returns:
      Fn_clean: (M, T) with first max_pcomps PCs subtracted (back in ~1 space)
      meta: dict
    """
    Fn_subset = np.asarray(Fn_subset, float)
    M, T = Fn_subset.shape

    # 1) sigma-clip outliers per star (optional but recommended)
    Fn_clip = sigma_clip_lightcurves(Fn_subset, sigma=sigma_clip_sigma, iters=sigma_clip_iters)

    # 2) impute NaNs with per-star median (avoid hard 1.0 fill artifacts)
    star_med = np.nanmedian(Fn_clip, axis=1, keepdims=True)
    star_med = np.where(np.isfinite(star_med), star_med, center_value)
    Fn_imp = np.where(np.isfinite(Fn_clip), Fn_clip, star_med)

    # 3) center around 0 for PCA (since curves are around 1)
    X = Fn_imp - center_value

    # 4) PCA
    k = int(min(max_pcomps, M, T))
    if k <= 0:
        return Fn_subset.copy(), {"n_components_used": 0}

    pca = PCA(n_components=k)
    pca.fit(X)

    # 5) reconstruct common-mode model and subtract
    X_model = pca.inverse_transform(pca.transform(X))
    X_clean = X - X_model

    # 6) uncenter back to ~1
    Fn_clean = X_clean + center_value

    meta = {
        "n_components_used": k,
        "explained_variance_ratio": getattr(pca, "explained_variance_ratio_", np.array([])),
    }
    return Fn_clean, meta


def compute_normalized_matrix(F, min_net_flux=0.0):
    """
    F: (N, T) raw flux matrix
    Returns Fn: normalized by per-star mean over finite > min_net_flux samples (mean ~ 1)
    """
    F = np.asarray(F, float)
    valid = np.isfinite(F) & (F > float(min_net_flux))
    means = np.nanmean(np.where(valid, F, np.nan), axis=1)
    means = np.where(np.isfinite(means) & (means > 0), means, np.nan)
    Fn = F / means[:, None]
    return Fn


def _pca_detrend_with_refs_only(Fn_refs, Fn_target, max_pcomps=3):
    """
    Fit PCA on reference normalized curves only, then remove the reconstructed
    common-mode from BOTH refs and target. Returns detrended (baseline~1) curves.

    Fn_refs   : (Nref, T)
    Fn_target : (T,)
    """
    Fn_refs = np.asarray(Fn_refs, float)
    Fn_target = np.asarray(Fn_target, float)

    nref, T = Fn_refs.shape
    if nref < 2 or T < 3:
        # Not enough to do PCA robustly; return originals
        return Fn_refs.copy(), Fn_target.copy()

    # Replace NaNs with 1.0 for PCA math (baseline for normalized curves)
    Xr = np.where(np.isfinite(Fn_refs), Fn_refs, 1.0)
    xt = np.where(np.isfinite(Fn_target), Fn_target, 1.0).reshape(1, -1)

    n_components = int(min(max_pcomps, nref, T))
    if n_components < 1:
        return Fn_refs.copy(), Fn_target.copy()

    pca = PCA(n_components=n_components)
    pca.fit(Xr)

    # Reconstruct common-mode
    Xr_common = pca.inverse_transform(pca.transform(Xr))
    xt_common = pca.inverse_transform(pca.transform(xt)).reshape(-1)

    # Detrend: subtract common-mode and re-add baseline 1.0
    Xr_det = Xr - Xr_common + 1.0
    xt_det = xt.reshape(-1) - xt_common + 1.0

    # Put NaNs back where originals were invalid (optional but keeps masking behavior)
    Xr_det = np.where(np.isfinite(Fn_refs), Xr_det, np.nan)
    xt_det = np.where(np.isfinite(Fn_target), xt_det, np.nan)

    return Xr_det, xt_det


def compute_rel_curve_pca_cleaned(
    F,
    target_i,
    ref_idx,
    min_net_flux=0.0,
    min_refs_per_frame=3,
    max_pcomps=2,
):
    """
    IDL-style: build normalized curves Fn, PCA-detrend using refs, then compute:
      ref_mean(t) = mean(detrended refs at t) using frames with >= min_refs_per_frame
      rel(t)      = detrended_target(t) / ref_mean(t)

    Returns:
      rel       : (T,)
      ref_mean  : (T,)
      Fn_t_det  : (T,)  detrended normalized target
      Fn_r_det  : (Nref, T) detrended normalized refs
    """
    F = np.asarray(F, float)
    N, T = F.shape

    # Valid mask for computing per-star mean (avoid zeros/negatives)
    valid = np.isfinite(F) & (F > float(min_net_flux))

    means = np.nanmean(np.where(valid, F, np.nan), axis=1)
    # Guard: avoid divide-by-zero
    means[~np.isfinite(means) | (means <= 0)] = np.nan

    Fn = F / means[:, None]

    Fn_t = Fn[target_i, :]
    Fn_r = Fn[np.asarray(ref_idx, int), :]

    # Optional: light sigma-clip (uses your existing helper)
    # Keep it mild so we don't destroy real events
    try:
        Fn_r = sigma_clip_lightcurves(Fn_r, sigma=4.0, iters=1)
        Fn_t = sigma_clip_lightcurves(Fn_t.reshape(1, -1), sigma=6.0, iters=1).reshape(-1)
    except Exception:
        pass

    # PCA detrend using refs only
    Fn_r_det, Fn_t_det = _pca_detrend_with_refs_only(Fn_r, Fn_t, max_pcomps=max_pcomps)

    # Build ref_mean per frame with min_refs_per_frame constraint
    ok_ref = np.isfinite(Fn_r_det)
    n_ok = ok_ref.sum(axis=0)

    ref_mean = np.full(T, np.nan, dtype=float)
    good_frames = n_ok >= int(min_refs_per_frame)
    if np.any(good_frames):
        ref_mean[good_frames] = np.nanmean(np.where(ok_ref, Fn_r_det, np.nan), axis=0)[good_frames]

    # Relative curve
    rel = np.full(T, np.nan, dtype=float)
    good = np.isfinite(Fn_t_det) & np.isfinite(ref_mean) & (ref_mean > 0)
    rel[good] = Fn_t_det[good] / ref_mean[good]

    return rel, ref_mean, Fn_t_det, Fn_r_det

def save_target_specific_lightcurves_idl_style_old(
    mags_tab,
    positions0,
    out_dir="lightcurves_idlstyle",
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_points=5,
    min_refs_per_frame=3,
    dpi=150,
    normalize_rel_by_median=True,
    normalize_ref_by_median=False,
    write_ref_lists=True,
    max_pcomps=2,   # NEW (safe default)
):
    """
    For each star:
      1) select refs (your existing selection function)
      2) compute PCA-cleaned relative curve (propagated to subtraction)
      3) save 3-panel plot: raw target flux, PCA-cleaned ref mean, PCA-cleaned relative

    Note: PCA here is NOT just for selecting refs; it affects ref_mean and rel curve.
    """
    os.makedirs(out_dir, exist_ok=True)

    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)
    N, T = F.shape

    # For optional ref list writing
    ref_list_dir = os.path.join(out_dir, "ref_lists")
    if write_ref_lists:
        os.makedirs(ref_list_dir, exist_ok=True)

    # Sort frames by JD for plotting
    jd_plot = np.asarray(jd_per_frame, float)
    t_order = np.argsort(jd_plot)

    n_saved = 0
    for target_i in range(N):
        sid = int(ids_sorted[target_i])

        # Need some data to plot
        ft = F[target_i, :]
        if np.isfinite(ft).sum() < int(min_points):
            continue

        ref_idx = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, min_refs_per_frame),
        )
        if len(ref_idx) < int(min_refs_per_frame):
            continue

        # --- PCA-propagated detrending + relative ---
        rel, ref_mean, Fn_t_det, Fn_r_det = compute_rel_curve_pca_cleaned(
            F=F,
            target_i=target_i,
            ref_idx=ref_idx,
            min_net_flux=min_net_flux,
            min_refs_per_frame=min_refs_per_frame,
            max_pcomps=max_pcomps,
        )

        # Optional write list of reference STAR IDS (not row indices)
        if write_ref_lists:
            out_list = os.path.join(ref_list_dir, f"refs_for_star_{sid:05d}.txt")
            with open(out_list, "w", encoding="utf-8") as f:
                f.write("# Reference star IDs (target-specific)\n")
                f.write(f"# target_id = {sid}\n")
                for ri in ref_idx:
                    f.write(f"{int(ids_sorted[int(ri)])}\n")

        # Prepare plot series
        raw_flux = ft.copy()

        ref_plot = ref_mean.copy()
        if normalize_ref_by_median:
            med = np.nanmedian(ref_plot)
            if np.isfinite(med) and med != 0:
                ref_plot = ref_plot / med

        rel_plot = rel.copy()
        if normalize_rel_by_median:
            med = np.nanmedian(rel_plot)
            if np.isfinite(med) and med != 0:
                rel_plot = rel_plot / med

        # Require enough points in rel to be worth saving
        if np.isfinite(rel_plot).sum() < int(min_points):
            continue

        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

        axes[0].plot(jd_plot[t_order], raw_flux[t_order], marker="o", linestyle="-", markersize=3)
        axes[0].set_ylabel(flux_col)
        axes[0].set_title(f"Star id={sid} — Raw target flux")

        axes[1].plot(jd_plot[t_order], ref_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[1].set_ylabel("ref_mean (PCA-cleaned)")
        axes[1].set_title("Reference light curve (PCA-cleaned mean of selected refs)")

        axes[2].plot(jd_plot[t_order], rel_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[2].set_ylabel("relative (target/ref)")
        axes[2].set_xlabel("JD")
        axes[2].set_title("Relative light curve" + (" (median-normalized)" if normalize_rel_by_median else ""))

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"star_{sid:05d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)
        n_saved += 1

    print(f"Saved {n_saved} IDL-style per-target 3-panel light-curve plots to: {out_dir}")


def save_target_specific_lightcurves_idl_style(
    mags_tab,
    positions0,
    out_dir="lightcurves_idlstyle",
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_points=5,
    min_refs_per_frame=3,
    dpi=150,
    normalize_rel_by_median=True,
    normalize_ref_by_median=False,
    write_ref_lists=True,
    max_pcomps=2,                 # tune this (1–3)
    save_pca_features=True,        # existing
    # ---------------- NEW: add a 4th panel with refs before/after PCA ----------------
    add_refs_panel=True,
    plot_refs_before=True,         # dotted
    plot_refs_after=True,          # solid
    max_refs_plot=12,              # limit how many refs are drawn in the 4th panel
    # ---------------- NEW: save per-target IDL/PCA photometric tables ----------------
    save_idl_tables=True,
    idl_tables_dirname="idl_pca_tables",
    save_csv=True,                 # <-- CSV by default (as requested)
    save_npz=False,                # optional compressed arrays
    max_refs_save=None,            # None = save all selected refs; or set e.g. 20 to cap
):
    """
    Saves per-target IDL-style plots and (optionally) per-target tables.

    Panels:
      1) target raw net_flux vs JD
      2) reference mean (PCA-cleaned) vs JD
      3) relative (target/ref) vs JD
      4) (optional) selected ref stars before PCA (dotted) and after PCA (solid)

    Tables (optional, per target):
      - CSV (wide): JD, target_raw, target_before_norm, target_after_norm, ref_mean_pca_clean, relative,
                   plus ref_{ID}_before_norm and ref_{ID}_after_norm columns for selected refs.
      - NPZ (optional): same data in array form.
    """
    import os
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)
    N, T = F.shape

    jd_plot = np.asarray(jd_per_frame, float)
    t_order = np.argsort(jd_plot)
    x = jd_plot[t_order]

    ref_list_dir = os.path.join(out_dir, "ref_lists")
    if write_ref_lists:
        os.makedirs(ref_list_dir, exist_ok=True)

    pca_dir = os.path.join(out_dir, "pca_features")
    if save_pca_features:
        os.makedirs(pca_dir, exist_ok=True)

    # ---- NEW: tables directory ----
    lc_data_dir = os.path.join(out_dir, idl_tables_dirname)
    if save_idl_tables:
        os.makedirs(lc_data_dir, exist_ok=True)

    # ---- NEW: precompute "before PCA" normalized matrix (same convention used in PCA routine) ----
    valid_all = np.isfinite(F) & (F > float(min_net_flux))
    means_all = np.nanmean(np.where(valid_all, F, np.nan), axis=1)
    means_all = np.where(np.isfinite(means_all) & (means_all > 0), means_all, np.nan)
    Fn_all = F / means_all[:, None]  # normalized ~1

    n_saved = 0
    for target_i, sid in enumerate(ids_sorted):
        ft = F[target_i, :]

        ok_t = np.isfinite(ft) & (ft > float(min_net_flux))
        if ok_t.sum() < int(min_points):
            continue

        ref_idx, stats = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, int(min_refs_per_frame)),
            return_stats=True,
        )
        if len(ref_idx) < int(min_refs_per_frame):
            continue

        # --- PCA AFTER final selection ---
        # returns: rel (T,), ref_mean (T,), Fn_t_clean (T,), Fn_r_clean (R,T), comps_TK (T,K), scores_MK (R,K), target_score (K,)
        rel, ref_mean, Fn_t_clean, Fn_r_clean, comps_TK, scores_MK, target_score = compute_rel_curve_pca_postselect(
            F=F,
            target_i=target_i,
            ref_idx=ref_idx,
            min_net_flux=min_net_flux,
            min_refs_per_frame=min_refs_per_frame,
            max_pcomps=max_pcomps,
            return_pca_features=True,
        )

        if np.sum(np.isfinite(rel) & (rel > 0)) < int(min_points):
            continue

        # optional normalization for display
        rel_plot = rel.copy()
        if normalize_rel_by_median:
            med = np.nanmedian(rel_plot[rel_plot > 0])
            if np.isfinite(med) and med > 0:
                rel_plot = rel_plot / med

        ref_plot = ref_mean.copy()
        if normalize_ref_by_median:
            medr = np.nanmedian(ref_plot[ref_plot > 0])
            if np.isfinite(medr) and medr > 0:
                ref_plot = ref_plot / medr

        # ---------- Save ref list ----------
        if write_ref_lists:
            ref_path = os.path.join(ref_list_dir, f"star_{int(sid):05d}_refs.txt")
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write("# target_id\n")
                f.write(f"{int(sid)}\n")
                f.write("# ref_ids\n")
                for rid in ids_sorted[ref_idx]:
                    f.write(f"{int(rid)}\n")

        # ---------- NEW: Save per-target IDL/PCA tables ----------
        if save_idl_tables:
            sid_int = int(sid)
            ref_ids_all = ids_sorted[np.asarray(ref_idx, int)].astype(int)

            # arrays (full length T)
            target_raw = ft.astype(float)
            target_before = Fn_all[target_i, :].astype(float)          # normalized (~1), before PCA
            target_after = np.asarray(Fn_t_clean, float)               # normalized (~1), after PCA
            ref_mean_clean = np.asarray(ref_mean, float)               # PCA-cleaned ref mean
            relative = np.asarray(rel, float)                          # target/ref (pre-median-norm)

            refs_before = Fn_all[np.asarray(ref_idx, int), :].astype(float)  # (R,T)
            refs_after = np.asarray(Fn_r_clean, float)                        # (R,T)

            # optionally cap how many refs we save (to keep CSV size reasonable)
            if max_refs_save is not None and refs_before.shape[0] > int(max_refs_save):
                # keep brightest selected refs by mean raw flux
                ref_flux_means = means_all[np.asarray(ref_idx, int)]
                keep = np.argsort(ref_flux_means)[::-1][: int(max_refs_save)]
                ref_ids = ref_ids_all[keep]
                refs_before = refs_before[keep, :]
                refs_after = refs_after[keep, :]
            else:
                ref_ids = ref_ids_all

            if save_npz:
                npz_path = os.path.join(lc_data_dir, f"star_{sid_int:05d}_idl_pca.npz")
                np.savez_compressed(
                    npz_path,
                    jd=jd_plot.astype(float),
                    t_order=t_order.astype(int),
                    target_id=sid_int,
                    ref_ids=ref_ids.astype(int),
                    target_raw=target_raw,
                    target_before_norm=target_before,
                    target_after_norm=target_after,
                    ref_mean_pca_clean=ref_mean_clean,
                    relative_target_over_ref=relative,
                    refs_before_norm=refs_before,
                    refs_after_norm=refs_after,
                    pca_components=comps_TK if comps_TK is not None else None,
                    pca_scores_refs=scores_MK if scores_MK is not None else None,
                    pca_score_target=target_score if target_score is not None else None,
                )

            if save_csv:
                csv_path = os.path.join(lc_data_dir, f"star_{sid_int:05d}_idl_pca.csv")

                # Build header
                header = [
                    "JD",
                    "target_raw",
                    "target_before_norm",
                    "target_after_norm",
                    "ref_mean_pca_clean",
                    "relative_target_over_ref",
                ]
                for rid in ref_ids:
                    header.append(f"ref_{int(rid)}_before_norm")
                for rid in ref_ids:
                    header.append(f"ref_{int(rid)}_after_norm")

                # Write rows
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    for t in range(T):
                        row = [
                            float(jd_plot[t]),
                            float(target_raw[t]) if np.isfinite(target_raw[t]) else "",
                            float(target_before[t]) if np.isfinite(target_before[t]) else "",
                            float(target_after[t]) if np.isfinite(target_after[t]) else "",
                            float(ref_mean_clean[t]) if np.isfinite(ref_mean_clean[t]) else "",
                            float(relative[t]) if np.isfinite(relative[t]) else "",
                        ]
                        # refs before
                        for j in range(len(ref_ids)):
                            v = refs_before[j, t]
                            row.append(float(v) if np.isfinite(v) else "")
                        # refs after
                        for j in range(len(ref_ids)):
                            v = refs_after[j, t]
                            row.append(float(v) if np.isfinite(v) else "")
                        w.writerow(row)

        # ---------- Plotting ----------
        nrows = 4 if add_refs_panel else 3
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 13 if add_refs_panel else 10), sharex=True)

        # FIX: x and y ordering must match
        axes[0].plot(x, ft[t_order], marker="o", linestyle="-", markersize=3)
        axes[0].set_ylabel("net_flux (target)")
        stats_str = f"Cand={stats['total_in_radius']}, Sel={stats['selected']}, Rej={stats['rejected']}"
        axes[0].set_title(f"Star id={int(sid)} ({stats_str})")

        axes[1].plot(x, ref_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[1].set_ylabel("ref_mean (PCA-cleaned)")
        axes[1].set_title("Reference light curve (PCA after final selection)")

        axes[2].plot(x, rel_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[2].set_ylabel("relative (target/ref)")
        axes[2].set_title("Relative light curve" + (" (median-normalized)" if normalize_rel_by_median else ""))

        if add_refs_panel:
            ax_refs = axes[3]

            # Limit number of refs drawn (for readability)
            ref_idx_plot = ref_idx
            if max_refs_plot is not None:
                ref_idx_plot = ref_idx[:max_refs_plot]

            for r_i in ref_idx_plot:
                r_i = int(r_i)

                # --- RAW BEFORE PCA (true net_flux) ---
                raw_before = F[r_i, :]

                if plot_refs_before:
                    ax_refs.plot(
                        x,
                        raw_before[t_order],
                        linestyle=":",
                        linewidth=1,
                        alpha=0.6
                    )

                # --- RAW AFTER PCA ---
                # Fn_r_clean is normalized, so convert back to flux
                mean_flux = means_all[r_i]
                raw_after = Fn_r_clean[
                    np.where(ref_idx == r_i)[0][0], :
                ] * mean_flux

                if plot_refs_after:
                    ax_refs.plot(
                        x,
                        raw_after[t_order],
                        linestyle="-",
                        linewidth=1.2,
                        alpha=0.9
                    )

            ax_refs.set_ylabel("net_flux")
            ax_refs.set_title("Selected reference stars (raw before PCA dotted, raw after PCA solid)")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"star_{int(sid):05d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)

        # ---------- PCA feature plot per star ----------
        if save_pca_features and comps_TK is not None and scores_MK is not None:
            ids_M = ids_sorted[ref_idx].astype(int)
            feat_path = os.path.join(pca_dir, f"star_{int(sid):05d}_pca.png")
            save_pca_feature_plot(
                feat_path,
                jd_plot[t_order],
                comps_TK[t_order, :],
                scores_MK,          # refs only
                ids_M=ids_M,        # refs only
                title=f"Star {int(sid)} PCA refs-only (K={scores_MK.shape[1]})",
            )

        n_saved += 1

    print(f"Saved {n_saved} IDL-style per-target plots to: {out_dir}")
    if save_idl_tables:
        print(f"Saved IDL/PCA tables to: {os.path.join(out_dir, idl_tables_dirname)}")


def save_target_specific_lightcurves_idl_style_old(
    mags_tab,
    positions0,
    out_dir="lightcurves_idlstyle",
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_points=5,
    min_refs_per_frame=3,
    dpi=150,
    normalize_rel_by_median=True,
    normalize_ref_by_median=False,
    write_ref_lists=True,
):
    """
    For EACH star (target):
      - select refs using IDL-like method (relative-to-target)
      - build reference light curve ref_mean(t) = mean(net_flux[refs, t]) with >= min_refs_per_frame refs
      - compute rel_flux(t) = net_flux_target(t) / ref_mean(t)
      - save a 3-panel plot:
            (1) raw target net_flux vs JD
            (2) reference ref_mean vs JD
            (3) relative rel_flux vs JD (optional median normalization)
    """
    os.makedirs(out_dir, exist_ok=True)
    ref_list_dir = os.path.join(out_dir, "ref_lists")
    if write_ref_lists:
        os.makedirs(ref_list_dir, exist_ok=True)

    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)

    # time order by JD if possible
    if np.any(np.isfinite(jd_per_frame)):
        t_order = np.argsort(np.where(np.isfinite(jd_per_frame), jd_per_frame, np.inf))
    else:
        t_order = np.arange(len(frames_sorted))
    jd_plot = jd_per_frame[t_order]

    n_saved = 0
    for target_i, sid in enumerate(ids_sorted):
        ft = F[target_i, :]

        ok_t = np.isfinite(ft) & (ft > min_net_flux)
        if ok_t.sum() < min_points:
            continue

        ref_idx, stats = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, min_refs_per_frame),
            return_stats=True,
        )
        if len(ref_idx) < min_refs_per_frame:
            continue

        # reference curve
        Fref = F[ref_idx, :]  # (R,T)
        ok_ref = np.isfinite(Fref) & (Fref > min_net_flux)
        n_ok = ok_ref.sum(axis=0)

        ref_mean = np.full(F.shape[1], np.nan, dtype=float)
        good_frames = n_ok >= min_refs_per_frame
        if np.any(good_frames):
            ref_mean[good_frames] = np.nanmean(np.where(ok_ref, Fref, np.nan), axis=0)[good_frames]

        # relative curve
        rel = np.full_like(ft, np.nan, dtype=float)
        good = np.isfinite(ft) & (ft > min_net_flux) & np.isfinite(ref_mean) & (ref_mean > 0)
        rel[good] = ft[good] / ref_mean[good]

        

        if np.sum(np.isfinite(rel) & (rel > 0)) < min_points:
            continue

        rel_plot = rel.copy()
        if normalize_rel_by_median:
            med = np.nanmedian(rel_plot[rel_plot > 0])
            if np.isfinite(med) and med > 0:
                rel_plot = rel_plot / med

        ref_plot = ref_mean.copy()
        if normalize_ref_by_median:
            medr = np.nanmedian(ref_plot[ref_plot > 0])
            if np.isfinite(medr) and medr > 0:
                ref_plot = ref_plot / medr

        if write_ref_lists:
            ref_path = os.path.join(ref_list_dir, f"star_{int(sid):05d}_refs.txt")
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write("# target_id\n")
                f.write(f"{int(sid)}\n")
                f.write("# ref_ids\n")
                for rid in ids_sorted[ref_idx]:
                    f.write(f"{int(rid)}\n")

        # 3-panel plot: raw + reference + relative
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axes[0].plot(jd_plot, ft[t_order], marker="o", linestyle="-", markersize=3)
        axes[0].set_ylabel("net_flux (target)")
        stats_str = f"Cand={stats['total_in_radius']}, Sel={stats['selected']}, Rej={stats['rejected']}"
        axes[0].set_title(f"Star id={int(sid)} ({stats_str})")

        axes[1].plot(jd_plot, ref_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[1].set_ylabel("ref_mean (refs)")
        axes[1].set_title("Reference light curve (mean of selected refs)")

        axes[2].plot(jd_plot, rel_plot[t_order], marker="o", linestyle="-", markersize=3)
        axes[2].set_ylabel("relative (target/ref)")
        axes[2].set_xlabel("JD")
        axes[2].set_title("Relative light curve" + (" (median-normalized)" if normalize_rel_by_median else ""))

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"star_{int(sid):05d}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close(fig)
        n_saved += 1

    print(f"Saved {n_saved} IDL-style per-target 3-panel light-curve plots to: {out_dir}")

def add_idl_relative_flux_columns(
    mags_tab,
    positions0,
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_refs_per_frame=3,
    normalize_rel_by_median=True,
    max_pcomps=2,   # PCA after final selection
):
    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)
    N, T = F.shape

    R = np.full((N, T), np.nan, dtype=float)

    for target_i in range(N):
        ref_idx = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, min_refs_per_frame),
        )
        if len(ref_idx) < int(min_refs_per_frame):
            continue

        rel, ref_mean, Fn_t_clean, Fn_r_clean = compute_rel_curve_pca_postselect(
            F=F,
            target_i=target_i,
            ref_idx=ref_idx,
            min_net_flux=min_net_flux,
            min_refs_per_frame=min_refs_per_frame,
            max_pcomps=max_pcomps,
            return_pca_features=False,
        )
        R[target_i, :] = rel

    id_to_i = {int(sid): i for i, sid in enumerate(ids_sorted)}
    fr_to_t = {int(fr): t for t, fr in enumerate(frames_sorted)}

    rel_idl = np.full(len(mags_tab), np.nan, dtype=float)
    for k, row in enumerate(mags_tab):
        sid = int(row["id"])
        fr = int(row["frame"])
        i = id_to_i.get(sid, None)
        t = fr_to_t.get(fr, None)
        if i is None or t is None:
            continue
        rel_idl[k] = R[i, t]

    mags2 = mags_tab.copy()
    mags2["rel_flux_idl"] = rel_idl

    if normalize_rel_by_median:
        mags2 = normalize_rel_flux_per_star(
            mags2, id_col="id", rel_flux_col="rel_flux_idl", out_col="rel_flux_idl_norm"
        )

    return mags2


def add_idl_relative_flux_columns_old_second(
    mags_tab,
    positions0,
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_refs_per_frame=3,
    normalize_rel_by_median=True,
    max_pcomps=2,   # NEW (safe default)
):
    """
    Computes IDL-style *target-specific* relative flux for EVERY star and
    attaches it to mags_tab as:
      - rel_flux_idl         (PCA-cleaned relative curve, propagated)
      - rel_flux_idl_norm    (optional median-normalized per star)

    This replaces the straight-mean ref subtraction in the current implementation
    :contentReference[oaicite:4]{index=4} with PCA-propagated detrending.
    """
    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)
    N, T = F.shape

    R = np.full((N, T), np.nan, dtype=float)

    for target_i in range(N):
        ref_idx = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, min_refs_per_frame),
        )
        if len(ref_idx) < int(min_refs_per_frame):
            continue

        rel, ref_mean, Fn_t_det, Fn_r_det = compute_rel_curve_pca_cleaned(
            F=F,
            target_i=target_i,
            ref_idx=ref_idx,
            min_net_flux=min_net_flux,
            min_refs_per_frame=min_refs_per_frame,
            max_pcomps=max_pcomps,
        )
        R[target_i, :] = rel

    # Map back to table rows
    id_to_i = {int(sid): i for i, sid in enumerate(ids_sorted)}
    fr_to_t = {int(fr): t for t, fr in enumerate(frames_sorted)}

    rel_idl = np.full(len(mags_tab), np.nan, dtype=float)
    for k, row in enumerate(mags_tab):
        sid = int(row["id"])
        fr = int(row["frame"])
        i = id_to_i.get(sid, None)
        t = fr_to_t.get(fr, None)
        if i is None or t is None:
            continue
        rel_idl[k] = R[i, t]

    mags2 = mags_tab.copy()
    mags2["rel_flux_idl"] = rel_idl

    if normalize_rel_by_median:
        mags2 = normalize_rel_flux_per_star(
            mags2, id_col="id", rel_flux_col="rel_flux_idl", out_col="rel_flux_idl_norm"
        )

    return mags2


# ----------------------------
# NEW: compute IDL relative flux column for ALL rows (for SNR)
# ----------------------------

def add_idl_relative_flux_columns_old(
    mags_tab,
    positions0,
    flux_col="net_flux",
    comp_rad_pix=100.0,
    fraction_keep_pca=0.95,
    fraction_keep_comps=0.25,
    min_frac_present=1.0,
    min_net_flux=0.0,
    min_refs_per_frame=3,
    normalize_rel_by_median=True,
):
    """
    Computes IDL-style *target-specific* relative flux for EVERY star and
    attaches it to mags_tab as:
      - rel_flux_idl
      - rel_flux_idl_norm (optional median-normalized per star)
    """
    ids_sorted, frames_sorted, jd_per_frame, F = build_flux_matrix(mags_tab, flux_col=flux_col)

    N, T = F.shape
    R = np.full((N, T), np.nan, dtype=float)

    for target_i in range(N):
        ft = F[target_i, :]

        ref_idx = select_refs_idl_relative_for_target(
            target_i=target_i,
            F=F,
            positions0=positions0,
            comp_rad_pix=comp_rad_pix,
            fraction_keep_pca=fraction_keep_pca,
            fraction_keep_comps=fraction_keep_comps,
            min_frac_present=min_frac_present,
            min_net_flux=min_net_flux,
            min_keep=max(3, min_refs_per_frame),
        )
        if len(ref_idx) < min_refs_per_frame:
            continue

        Fref = F[ref_idx, :]
        ok_ref = np.isfinite(Fref) & (Fref > min_net_flux)
        n_ok = ok_ref.sum(axis=0)

        ref_mean = np.full(T, np.nan, dtype=float)
        good_frames = n_ok >= min_refs_per_frame
        if np.any(good_frames):
            ref_mean[good_frames] = np.nanmean(np.where(ok_ref, Fref, np.nan), axis=0)[good_frames]

        good = np.isfinite(ft) & (ft > min_net_flux) & np.isfinite(ref_mean) & (ref_mean > 0)
        R[target_i, good] = ft[good] / ref_mean[good]

    id_to_i = {int(sid): i for i, sid in enumerate(ids_sorted)}
    fr_to_t = {int(fr): t for t, fr in enumerate(frames_sorted)}

    rel_idl = np.full(len(mags_tab), np.nan, dtype=float)
    for k, row in enumerate(mags_tab):
        sid = int(row["id"])
        fr = int(row["frame"])
        i = id_to_i.get(sid, None)
        t = fr_to_t.get(fr, None)
        if i is None or t is None:
            continue
        rel_idl[k] = R[i, t]

    mags2 = mags_tab.copy()
    mags2["rel_flux_idl"] = rel_idl

    if normalize_rel_by_median:
        mags2 = normalize_rel_flux_per_star(
            mags2, id_col="id", rel_flux_col="rel_flux_idl", out_col="rel_flux_idl_norm"
        )

    return mags2




# ----------------------------
# ONLINE ASTROMETRY + GAIA
# ----------------------------

def get_astrometry_key():
    k = (ASTROMETRY_API_KEY or "").strip()
    if k:
        return k
    return os.environ.get("ASTROMETRY_NET_API_KEY", "").strip()


def solve_wcs_online_astrometrynet(fits_path, api_key):
    if not api_key:
        raise RuntimeError(
            "No astrometry.net API key provided. Set ASTROMETRY_NET_API_KEY (env var)."
        )

    ast = AstrometryNet()
    ast.api_key = api_key

    pix_scale = 0.34   # arcsec/pixel (adjust if needed)
    radius_deg = 0.16  # half-width of FoV (adjust if needed)

    settings = dict(
        publicly_visible="n",
        downsample_factor=1,
        tweak_order=1,
        scale_units="arcsecperpix",
        scale_type="ul",
        scale_lower=0.9 * pix_scale,
        scale_upper=1.1 * pix_scale,
        radius=float(radius_deg),
    )

    try:
        hdr_wcs = ast.solve_from_image(fits_path, **settings)
    except TypeError:
        hdr_wcs = ast.solve_from_image(fits_path)

    if not hdr_wcs:
        raise RuntimeError("Astrometry.net solve failed (empty header returned).")

    return hdr_wcs


def wcs_star_radec_from_first_frame(wcs_header, xy_positions_first):
    w = WCS(wcs_header)
    xy = np.asarray(xy_positions_first, float)
    ok = np.isfinite(xy).all(axis=1)

    ra = np.full(len(xy), np.nan, dtype=float)
    dec = np.full(len(xy), np.nan, dtype=float)

    if np.any(ok):
        ra_ok, dec_ok = w.all_pix2world(xy[ok, 0], xy[ok, 1], 0)
        ra[ok] = ra_ok
        dec[ok] = dec_ok
    return ra, dec


def query_gaia_for_field(wcs_header, image_shape, extra_radius_factor=1.3, gmag_max=17.0):
    w = WCS(wcs_header)

    ny, nx = image_shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    ra_c, dec_c = w.all_pix2world(cx, cy, 0)

    scales = proj_plane_pixel_scales(w)  # deg/pix
    scale_deg = float(np.max(scales))
    diag_pix = np.sqrt((nx - 1) ** 2 + (ny - 1) ** 2)
    radius_deg = 0.5 * diag_pix * scale_deg * float(extra_radius_factor)

    Gaia.ROW_LIMIT = -1

    query = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_c}, {dec_c}, {radius_deg})
    )
    AND phot_g_mean_mag < {gmag_max}
    """

    job = Gaia.launch_job_async(query)
    tab = job.get_results()
    return tab


def crossmatch_to_gaia_old(star_ra_deg, star_dec_deg, gaia_tab, match_radius_arcsec=1.0):
    stars = SkyCoord(ra=star_ra_deg * u.deg, dec=star_dec_deg * u.deg, frame="icrs")
    cat = SkyCoord(ra=np.array(gaia_tab["ra"], float) * u.deg,
                   dec=np.array(gaia_tab["dec"], float) * u.deg,
                   frame="icrs")

    idx, d2d, _ = stars.match_to_catalog_sky(cat)
    match_radius = match_radius_arcsec * u.arcsec

    ok = np.isfinite(star_ra_deg) & np.isfinite(star_dec_deg) & (d2d < match_radius)

    gmag = np.full(len(star_ra_deg), np.nan, dtype=float)
    sid = np.full(len(star_ra_deg), -1, dtype=np.int64)
    sep = np.full(len(star_ra_deg), np.nan, dtype=float)

    if "phot_g_mean_mag" in gaia_tab.colnames:
        gmag_all = np.array(gaia_tab["phot_g_mean_mag"], float)
        gmag[ok] = gmag_all[idx[ok]]

    if "source_id" in gaia_tab.colnames:
        sid_all = np.array(gaia_tab["source_id"], np.int64)
        sid[ok] = sid_all[idx[ok]]

    sep[ok] = d2d[ok].to_value(u.arcsec)
    return gmag, sid, sep

# def crossmatch_to_gaia(star_ra_deg, star_dec_deg, gaia_tab, match_radius_arcsec=3.0):
#     """
#     Improved crossmatch: Finds the BRIGHTEST Gaia star within the radius
#     to avoid mismatching with deep background stars.
#     """
#     stars = SkyCoord(ra=star_ra_deg * u.deg, dec=star_dec_deg * u.deg, frame="icrs")
    
#     # 1. Sort Gaia table by magnitude (brightest first)
#     # This ensures the 'match_to_catalog_sky' prefers the primary star
#     gaia_tab.sort('phot_g_mean_mag')
    
#     cat = SkyCoord(ra=np.array(gaia_tab["ra"], float) * u.deg,
#                    dec=np.array(gaia_tab["dec"], float) * u.deg,
#                    frame="icrs")

#     # Match coordinates
#     idx, d2d, _ = stars.match_to_catalog_sky(cat)
#     match_radius = match_radius_arcsec * u.arcsec

#     ok = np.isfinite(star_ra_deg) & np.isfinite(star_dec_deg) & (d2d < match_radius)

#     gmag = np.full(len(star_ra_deg), np.nan, dtype=float)
#     sid = np.full(len(star_ra_deg), -1, dtype=np.int64)
#     sep = np.full(len(star_ra_deg), np.nan, dtype=float)
    
#     # Store Gaia RA/Dec for residuals diagnostic
#     gaia_ra = np.full(len(star_ra_deg), np.nan, dtype=float)
#     gaia_dec = np.full(len(star_ra_deg), np.nan, dtype=float)

#     if "phot_g_mean_mag" in gaia_tab.colnames:
#         gmag[ok] = gaia_tab["phot_g_mean_mag"][idx[ok]]
#         gaia_ra[ok] = gaia_tab["ra"][idx[ok]]
#         gaia_dec[ok] = gaia_tab["dec"][idx[ok]]

#     if "source_id" in gaia_tab.colnames:
#         sid[ok] = gaia_tab["source_id"][idx[ok]]

#     sep[ok] = d2d[ok].to_value(u.arcsec)
    
#     return gmag, sid, sep, gaia_ra, gaia_dec

def crossmatch_to_gaia_brightest(star_ra_deg, star_dec_deg, gaia_tab, match_radius_arcsec=1.0):
    """
    For each detected star, choose the BRIGHTEST Gaia source within match_radius_arcsec.
    Tie-breaker (optional): if equal brightness, choose the nearest.
    Returns: gmag, source_id, sep_arcsec, gaia_ra, gaia_dec
    """
    stars = SkyCoord(ra=np.array(star_ra_deg, float) * u.deg,
                     dec=np.array(star_dec_deg, float) * u.deg,
                     frame="icrs")

    cat = SkyCoord(ra=np.array(gaia_tab["ra"], float) * u.deg,
                   dec=np.array(gaia_tab["dec"], float) * u.deg,
                   frame="icrs")

    # all pairs within radius
    radius = float(match_radius_arcsec) * u.arcsec
    idx_star, idx_cat, d2d, _ = stars.search_around_sky(cat, radius)

    n = len(star_ra_deg)
    gmag = np.full(n, np.nan, dtype=float)
    sid  = np.full(n, -1, dtype=np.int64)
    sep  = np.full(n, np.nan, dtype=float)
    gaia_ra  = np.full(n, np.nan, dtype=float)
    gaia_dec = np.full(n, np.nan, dtype=float)

    if len(idx_star) == 0:
        return gmag, sid, sep, gaia_ra, gaia_dec

    # group candidates per star
    g_all = np.array(gaia_tab["phot_g_mean_mag"], float)
    sid_all = np.array(gaia_tab["source_id"], np.int64) if "source_id" in gaia_tab.colnames else None

    # For each star, pick candidate with minimum Gmag (brightest).
    # If you want a tie-breaker by distance, sort by (gmag, dist).
    for i in np.unique(idx_cat):
        pass  # (unused; keeping loop below by star index)

    for s_idx in np.unique(idx_cat * 0 + idx_star):  # unique star indices that have candidates
        m = (idx_star == s_idx)
        cands = idx_cat[m]
        dists = d2d[m].to_value(u.arcsec)
        mags  = g_all[cands]

        # ignore NaN mags if any
        okm = np.isfinite(mags)
        if not np.any(okm):
            continue

        cands = cands[okm]
        dists = dists[okm]
        mags  = mags[okm]

        # choose brightest, then nearest as tie-breaker
        j = np.lexsort((dists, mags))[0]  # sorts by mags primary, dists secondary
        best = cands[j]

        gmag[s_idx] = g_all[best]
        if sid_all is not None:
            sid[s_idx] = sid_all[best]
        sep[s_idx] = float(dists[j])
        gaia_ra[s_idx]  = float(gaia_tab["ra"][best])
        gaia_dec[s_idx] = float(gaia_tab["dec"][best])

    return gmag, sid, sep, gaia_ra, gaia_dec


# def build_star_meta_table(ids, x0, y0, ra_deg, dec_deg, gaia_gmag, gaia_source_id, gaia_sep_arcsec):
#     return Table(
#         rows=[
#             (int(i), float(x0[i]), float(y0[i]),
#              float(ra_deg[i]) if np.isfinite(ra_deg[i]) else np.nan,
#              float(dec_deg[i]) if np.isfinite(dec_deg[i]) else np.nan,
#              float(gaia_gmag[i]) if np.isfinite(gaia_gmag[i]) else np.nan,
#              int(gaia_source_id[i]),
#              float(gaia_sep_arcsec[i]) if np.isfinite(gaia_sep_arcsec[i]) else np.nan)
#             for i in ids
#         ],
#         names=["id", "x0", "y0", "RA_deg", "DEC_deg", "Gaia_Gmag", "Gaia_source_id", "Gaia_sep_arcsec"]
#     )

def build_star_meta_table(ids, x0, y0, ra_deg, dec_deg, gaia_gmag, gaia_source_id, gaia_sep_arcsec):
    # force arrays
    ids = np.asarray(ids, dtype=int)
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    gaia_gmag = np.asarray(gaia_gmag, dtype=float)
    gaia_source_id = np.asarray(gaia_source_id, dtype=np.int64)
    gaia_sep_arcsec = np.asarray(gaia_sep_arcsec, dtype=float)

    # hard sanity check (better error message than "index out of bounds")
    n = len(ids)
    lens = {
        "x0": len(x0), "y0": len(y0), "ra_deg": len(ra_deg), "dec_deg": len(dec_deg),
        "gaia_gmag": len(gaia_gmag), "gaia_source_id": len(gaia_source_id), "gaia_sep": len(gaia_sep_arcsec)
    }
    if any(v != n for v in lens.values()):
        raise RuntimeError(f"Meta length mismatch: n(ids)={n}, lengths={lens}")

    return Table(
        {
            "id": ids,
            "x0": x0,
            "y0": y0,
            "RA_deg": ra_deg,
            "DEC_deg": dec_deg,
            "Gaia_Gmag": gaia_gmag,
            "Gaia_source_id": gaia_source_id,
            "Gaia_sep_arcsec": gaia_sep_arcsec,
        }
    )


def attach_meta_to_table(tab, meta_tab):
    return join(tab, meta_tab, keys="id", join_type="left")


# ----------------------------
# SNR vs Gaia G
# ----------------------------

def snr_vs_gaia_gmag(mags_tab_with_meta,
                     n_first_frames=108,
                     flux_col="rel_flux",
                     gaia_mag_col="Gaia_Gmag",
                     out_txt="snr_vs_gaia_gmag.txt",
                     out_png="snr_vs_gaia_gmag.png",
                     min_points=5):
    frames = np.array(mags_tab_with_meta["frame"], dtype=int)
    jds = np.array(mags_tab_with_meta["JD"], dtype=float)

    frame_to_jd = {}
    for fr, jd in zip(frames, jds):
        if fr not in frame_to_jd and np.isfinite(jd):
            frame_to_jd[fr] = jd

    if frame_to_jd:
        frames_sorted = sorted(frame_to_jd.keys(), key=lambda f: frame_to_jd[f])
    else:
        frames_sorted = sorted(np.unique(frames))

    if n_first_frames is None:
        first_frames = np.array(frames_sorted, dtype=int)
        n_label = len(first_frames)
    else:
        first_frames = np.array(frames_sorted[:int(n_first_frames)], dtype=int)
        n_label = int(n_first_frames)

    tab = mags_tab_with_meta[np.isin(frames, first_frames)]
    ids = np.unique(np.array(tab["id"], dtype=int))

    rows = []
    xs, ys = [], []

    for sid in ids:
        r = tab[np.array(tab["id"], dtype=int) == sid]

        g = np.array(r[gaia_mag_col], dtype=float)
        g = g[np.isfinite(g)]
        if len(g) == 0:
            continue
        gmag = float(g[0])

        if flux_col not in r.colnames:
            continue

        f = np.array(r[flux_col], dtype=float)
        ok = np.isfinite(f) & (f > 0)
        f = f[ok]
        if len(f) < min_points:
            continue

        med = float(np.median(f))

        # Calculate Median Absolute Deviation
        abs_diff = np.abs(f - med)
        mad = np.median(abs_diff)
        
        # Convert MAD to a robust standard deviation equivalent
        # 1.4826 is the scaling factor for a normal distribution
        std = float(1.4826 * mad)
        
        # Ensure we don't divide by zero if the light curve is perfectly flat
        if not np.isfinite(std) or std <= 0:
            continue


        #std = float(np.std(f, ddof=1))
        #if not np.isfinite(std) or std <= 0:
        #    continue

        snr = med / std
        rows.append((int(sid), gmag, snr, med, std, int(len(f))))
        xs.append(gmag)
        ys.append(snr)

    snr_tab = Table(rows=rows, names=["id", "Gaia_Gmag", "snr", "median_flux", "std_flux", "npts"])
    snr_tab.write(out_txt, format="ascii.tab", overwrite=True)
    print(f"Wrote SNR table -> {out_txt}")

    xs = np.array(xs, float)
    ys = np.array(ys, float)

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=10)
    plt.xlabel("Gaia G magnitude")
    plt.ylabel(f"SNR = median({flux_col}) / std({flux_col})")
    plt.title(f"SNR vs Gaia G (first {n_label} frames)")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot -> {out_png}")

    return snr_tab


def diagnostic_1_visual_overlay(image, wcs_header, gaia_tab, star_positions, out_png="diag_1_overlay.png"):
    """Is the WCS correct? Checks if Gaia stars land on top of your image stars."""
    w = WCS(wcs_header)
    gaia_coords = SkyCoord(ra=gaia_tab['ra'], dec=gaia_tab['dec'], unit='deg', frame='icrs')
    gaia_px = w.world_to_pixel(gaia_coords)
    
    plt.figure(figsize=(10, 10))
    norm = simple_norm(image, 'sqrt', percent=99.5)
    plt.imshow(image, norm=norm, cmap='Greys_r')
    
    # Plot Detections (Red circles)
    plt.scatter(star_positions[:,0], star_positions[:,1], facecolors='none', 
                edgecolors='r', s=80, label='My Detections', alpha=0.6)
    # Plot Gaia (Cyan crosses)
    plt.scatter(gaia_px[0], gaia_px[1], marker='+', color='cyan', s=40, label='Gaia Sources', alpha=0.8)
    
    plt.title("Diagnostic 1: WCS Alignment (Red=Detect, Cyan=Gaia)")
    plt.legend()
    plt.savefig(out_png, dpi=200)
    plt.close()

def diagnostic_2_mag_correlation(mags_with_meta, out_png="diag_2_mag_linear.png"):
    """Is the match correct? Checks if Gaia Mag vs Instrumental Mag is a straight line."""
    # Filter for matched stars only
    matched = mags_with_meta[np.isfinite(mags_with_meta['Gaia_Gmag'])]
    if len(matched) == 0: return

    # Get median instrumental mag per star
    ids = np.unique(matched['id'])
    g_mags, i_mags = [], []
    for sid in ids:
        rows = matched[matched['id'] == sid]
        g_mags.append(rows['Gaia_Gmag'][0])
        i_mags.append(np.nanmedian(rows['mag_inst']))

    plt.figure(figsize=(8, 6))
    plt.scatter(g_mags, i_mags, alpha=0.5, s=15)
    plt.xlabel("Gaia G Magnitude")
    plt.ylabel("Instrumental Magnitude (-2.5*log10(Flux))")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title("Diagnostic 2: Photometric Correlation (Should be linear)")
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150)
    plt.close()

def diagnostic_3_residuals(mags_with_meta, out_png="diag_3_residuals.png"):
    """Is there a coordinate shift? Plots Delta-RA vs Delta-Dec in arcseconds."""
    matched = mags_with_meta[np.isfinite(mags_with_meta['Gaia_Gmag'])]
    if len(matched) == 0: return

    # Get unique star metadata
    _, unique_idx = np.unique(matched['id'], return_index=True)
    stars = matched[unique_idx]
    
    # Calculate residuals in arcseconds (approx)
    d_ra = (stars['RA_deg'] - stars['Gaia_RA']) * 3600 * np.cos(np.deg2rad(stars['DEC_deg']))
    d_dec = (stars['DEC_deg'] - stars['Gaia_Dec']) * 3600

    plt.figure(figsize=(6, 6))
    plt.scatter(d_ra, d_dec, alpha=0.6, s=20)
    plt.axhline(0, color='black', lw=0.5); plt.axvline(0, color='black', lw=0.5)
    plt.xlabel("RA Residual (arcsec)"); plt.ylabel("Dec Residual (arcsec)")
    plt.title("Diagnostic 3: Astrometric Residuals")
    plt.axis('equal')
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    data_directory = get_value("Enter the directory path for data images: ", DEV_DEFAULTS["data_directory"], str)
    dark_directory = get_value("Enter the directory path for dark images: ", DEV_DEFAULTS["dark_directory"], str)

    master_dark_fits = DEV_DEFAULTS["master_dark_fits"]
    median_fits = get_value("Enter median output FITS name: ", DEV_DEFAULTS["median_fits"], str)

    threshold = get_value("Enter threshold sigma for detection/tracking: ", DEV_DEFAULTS["threshold"], float)
    fwhm = get_value("Enter DAOStarFinder FWHM in pixels: ", DEV_DEFAULTS["fwhm"], float)

    box_size = get_value("Enter tracking box_size (odd int): ", DEV_DEFAULTS["box_size"], int)
    if box_size % 2 == 0:
        box_size += 1
        print(f"Made box_size odd -> {box_size}")

    max_match_dist_pix = float(DEV_DEFAULTS["max_match_dist_pix"])

    use_sharpness_filter = bool(DEV_DEFAULTS["use_sharpness_filter"])
    sharpness_min = float(DEV_DEFAULTS["sharpness_min"])
    sharpness_max = float(DEV_DEFAULTS["sharpness_max"])

    compare_jpeg = get_value("Enter JPEG name for side-by-side first/last: ", DEV_DEFAULTS["compare_jpeg"], str)

    coords_txt = DEV_DEFAULTS["coords_txt"]
    mags_txt = DEV_DEFAULTS["mags_txt"]
    mags_with_ref_and_diff_txt = DEV_DEFAULTS["mags_with_ref_and_diff_txt"]
    ref_ids_txt = DEV_DEFAULTS["ref_ids_txt"]
    ref_flux_txt = DEV_DEFAULTS["ref_flux_txt"]

    ref_lc_flux_png = DEV_DEFAULTS["ref_lc_flux_png"]
    ref_lc_mag_png = DEV_DEFAULTS["ref_lc_mag_png"]

    aperture_r = DEV_DEFAULTS["aperture_r"]
    sky_r_in = DEV_DEFAULTS["sky_r_in"]
    sky_r_out = DEV_DEFAULTS["sky_r_out"]

    min_frac_present = float(DEV_DEFAULTS["min_frac_present"])
    n_ref = DEV_DEFAULTS["n_ref"]
    min_net_flux = float(DEV_DEFAULTS["min_net_flux"])

    lc_out_dir = DEV_DEFAULTS["lc_out_dir"]
    lc_min_points = int(DEV_DEFAULTS["lc_min_points"])
    lc_dpi = int(DEV_DEFAULTS["lc_dpi"])

    snr_first_n = DEV_DEFAULTS["snr_first_n_frames"]  # int or None
    snr_flux_col = DEV_DEFAULTS["snr_flux_col"]
    snr_out_txt = DEV_DEFAULTS["snr_out_txt"]
    snr_out_png = DEV_DEFAULTS["snr_out_png"]

    gaia_match_radius = float(DEV_DEFAULTS["gaia_match_radius_arcsec"])
    gaia_extra_factor = float(DEV_DEFAULTS["gaia_extra_radius_factor"])

    # IDL-style per-target plots + IDL relative columns
    idl_out_dir = DEV_DEFAULTS["idl_out_dir"]
    idl_comp_rad_pix = float(DEV_DEFAULTS["idl_comp_rad_pix"])
    idl_fraction_keep_pca = float(DEV_DEFAULTS["idl_fraction_keep_pca"])
    idl_fraction_keep_comps = float(DEV_DEFAULTS["idl_fraction_keep_comps"])
    idl_min_refs_per_frame = int(DEV_DEFAULTS["idl_min_refs_per_frame"])
    idl_write_ref_lists = bool(DEV_DEFAULTS["idl_write_ref_lists"])
    idl_norm_rel = bool(DEV_DEFAULTS["idl_normalize_rel_by_median"])
    idl_norm_ref = bool(DEV_DEFAULTS["idl_normalize_ref_by_median"])

    # 1) Process images
    subtracted_images, data_headers, median_image, data_files = process_photometry_data(
        data_directory, dark_directory, median_fits, master_dark_fits
    )

    first_frame = subtracted_images[0]
    last_frame = subtracted_images[-1]

    # 2) Detect on first frame + initial tracking metadata
    star_positions_0, flux0, axis_ratio, peak0 = find_stars_first_frame_with_flux(
        first_frame, 
        threshold_sigma=threshold, 
        fwhm=fwhm,
        margin=10
    )

    # 3) Track positions across all frames
    
    #positions_per_frame = track_positions_refine_fluxmatch(
    #    subtracted_images,
    #    star_positions_0,
    #    flux0,
    #    axis_ratio=axis_ratio,  # From frame 0
    #    peak0=peak0,            # From frame 0
    #    box_size=box_size,
    #    threshold_sigma=threshold,
    #    fwhm=fwhm,
    #    max_match_dist_pix=1,
    #   ellip_max=0.2,         # Roundness limit
    #   guide_flux_min=1000,    # Adjust: Cutoff for faint stars
    #   guide_peak_max=10000    # Adjust: Cutoff for saturated stars (max pixel value)
    #

    positions_per_frame, shifts = track_positions_rigid_registration(
    subtracted_images,
    star_positions_0
    )


    #save_tracking_debug_images(subtracted_images, positions_per_frame, out_dir="tracked_images_circled")

    # 4) Photometry (writes coords_txt, mags_txt)
    coords_tab, mags_tab = aperture_photometry_sequence(
        subtracted_images, data_headers, positions_per_frame,
        aperture_r=aperture_r, sky_r_in=sky_r_in, sky_r_out=sky_r_out,
        coords_out=coords_txt, mags_out=mags_txt
    )

    # 4.25) NEW: compute IDL target-specific relative flux columns for ALL rows (used by SNR if selected)
    mags_tab = add_idl_relative_flux_columns(
        mags_tab=mags_tab,
        positions0=star_positions_0,
        flux_col="net_flux",
        comp_rad_pix=idl_comp_rad_pix,
        fraction_keep_pca=idl_fraction_keep_pca,
        fraction_keep_comps=idl_fraction_keep_comps,
        min_frac_present=min_frac_present,
        min_net_flux=min_net_flux,
        min_refs_per_frame=idl_min_refs_per_frame,
        normalize_rel_by_median=True,   # creates rel_flux_idl_norm
    )

    # 4.5) IDL-style per-target reference selection FOR EVERY STAR + 3-panel plots
    save_target_specific_lightcurves_idl_style(
        mags_tab=mags_tab,
        positions0=star_positions_0,
        out_dir=idl_out_dir,
        flux_col="net_flux",
        comp_rad_pix=idl_comp_rad_pix,
        fraction_keep_pca=idl_fraction_keep_pca,
        fraction_keep_comps=idl_fraction_keep_comps,
        min_frac_present=min_frac_present,
        min_net_flux=min_net_flux,
        min_points=lc_min_points,
        min_refs_per_frame=idl_min_refs_per_frame,
        dpi=lc_dpi,
        normalize_rel_by_median=idl_norm_rel,
        normalize_ref_by_median=idl_norm_ref,
        write_ref_lists=idl_write_ref_lists,
    )

    # 5) GLOBAL reference + differential (old pipeline still produced for comparison + your old outputs)
    #ref_ids = select_reference_star_ids(
    #    mags_tab, n_ref=n_ref, min_frac_present=min_frac_present, min_net_flux=min_net_flux
    #)
    #print(f"Using {len(ref_ids)} GLOBAL reference stars.")
    #write_reference_star_list(ref_ids, out_path=ref_ids_txt)

    #ref_flux_tab = compute_reference_flux_table(mags_tab, ref_ids, min_net_flux=min_net_flux)

    #print("Reference n_ref_used per frame:")
    #print("  min/median/max =",
    ##      np.min(ref_flux_tab["n_ref_used"]),
    #     np.median(ref_flux_tab["n_ref_used"]),
    #      np.max(ref_flux_tab["n_ref_used"]))

    #mags_diff = add_differential_columns(
    #    mags_tab, ref_flux_tab, ref_ids,
    #    denom_mode="mean",
    #    exclude_self_for_refstars=EXCLUDE_SELF_FOR_REF_STARS
    #)

    #mags_diff = normalize_rel_flux_per_star(mags_diff, out_col="rel_flux_norm")

    #ref_flux_tab.write(ref_flux_txt, format="ascii.tab", overwrite=True)
    #mags_diff.write(mags_with_ref_and_diff_txt, format="ascii.tab", overwrite=True)
    #print(f"Wrote differential photometry table -> {mags_with_ref_and_diff_txt}")

    plot_reference_lightcurve(ref_flux_tab, ref_lc_flux_png, ref_lc_mag_png, use="mean")

    save_first_last_side_by_side(
        first_frame, last_frame,
        positions_per_frame[0], positions_per_frame[-1],
        compare_jpeg, r=10.0, max_stars=None
    )

    # 5.5) Save per-star lightcurves using GLOBAL refs (your old plots)
    #save_lightcurves_for_all_stars_flux_only(
    #    mags_diff,
    #    out_dir=lc_out_dir,
    #    rel_flux_col="rel_flux_norm",
    #    raw_flux_col="net_flux",
    #    min_points=lc_min_points,
    #    dpi=lc_dpi
    #)

    # 6) Online WCS + Gaia + SNR vs Gaia G
    api_key = get_astrometry_key()
    first_fits_path = os.path.join(data_directory, data_files[0])
    print(f"Solving WCS online for: {first_fits_path}")


#old call
    try:
        wcs_header = solve_wcs_online_astrometrynet(first_fits_path, api_key=api_key)
        print("Astrometry.net solve OK. Doing Gaia match...")

        ids = np.arange(len(star_positions_0), dtype=int)
        ra_deg, dec_deg = wcs_star_radec_from_first_frame(wcs_header, star_positions_0)

        gaia_tab = query_gaia_for_field(wcs_header, first_frame.shape, extra_radius_factor=gaia_extra_factor)
        print(f"Gaia sources retrieved: {len(gaia_tab)}")

        gaia_gmag, gaia_source_id, gaia_sep = crossmatch_to_gaia_brightest(
            ra_deg, dec_deg, gaia_tab, match_radius_arcsec=gaia_match_radius
        )
        print(f"Gaia matches: {int(np.sum(np.isfinite(gaia_gmag)))}/{len(ids)}")

        meta_tab = build_star_meta_table(
            ids=ids,
            x0=star_positions_0[:, 0],
            y0=star_positions_0[:, 1],
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            gaia_gmag=gaia_gmag,
            gaia_source_id=gaia_source_id,
            gaia_sep_arcsec=gaia_sep
        )

        # Attach meta to coords, mags (with IDL columns), and mags_diff
        coords_with_meta = attach_meta_to_table(coords_tab, meta_tab)
        mags_with_meta = attach_meta_to_table(mags_tab, meta_tab)          # <-- includes rel_flux_idl(_norm)
        magsdiff_with_meta = attach_meta_to_table(mags_diff, meta_tab)

        coords_with_meta.write(coords_txt, format="ascii.tab", overwrite=True)
        mags_with_meta.write(mags_txt, format="ascii.tab", overwrite=True)
        magsdiff_with_meta.write(mags_with_ref_and_diff_txt, format="ascii.tab", overwrite=True)
        print("Re-wrote output tables with RA/Dec/Gaia columns.")

        # 7) SNR vs Gaia G
        # If snr_flux_col is IDL-based, it MUST be computed on mags_with_meta (not magsdiff_with_meta).
        # If you set snr_flux_col to "rel_flux" or "rel_flux_norm" (global), you can switch to magsdiff_with_meta.
        tab_for_snr = mags_with_meta if snr_flux_col.startswith("rel_flux_idl") else magsdiff_with_meta

        snr_vs_gaia_gmag(
            tab_for_snr,
            n_first_frames=snr_first_n,
            flux_col=snr_flux_col,
            gaia_mag_col="Gaia_Gmag",
            out_txt=snr_out_txt,
            out_png=snr_out_png,
            min_points=5
        )

    except Exception as e:
        print("WARNING: Online astrometry/Gaia step failed.")
        print(f"  Reason: {e}")
        print("  Your photometry outputs WITHOUT RA/Dec/Gaia were still produced.")


    # try:
    #     # --- A. SOLVE WCS ---
    #     wcs_header = solve_wcs_online_astrometrynet(first_fits_path, api_key=api_key)
    #     print("Astrometry.net solve OK. Doing Gaia crossmatch...")

    #     ids = np.arange(len(star_positions_0), dtype=int)
    #     ra_deg, dec_deg = wcs_star_radec_from_first_frame(wcs_header, star_positions_0)

    #     # --- B. QUERY GAIA ---
    #     gaia_tab = query_gaia_for_field(wcs_header, first_frame.shape, extra_radius_factor=gaia_extra_factor)
    #     print(f"Gaia sources retrieved: {len(gaia_tab)}")

    #     # --- C. IMPROVED CROSSMATCH (DIAGNOSTIC LOGIC FIX) ---
    #     # We sort by brightness to ensure we match the primary star, not background noise
    #     gaia_tab.sort('phot_g_mean_mag')
        
    #     stars_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    #     gaia_coords = SkyCoord(ra=np.array(gaia_tab["ra"]) * u.deg, 
    #                            dec=np.array(gaia_tab["dec"]) * u.deg, frame="icrs")

    #     idx, d2d, _ = stars_coords.match_to_catalog_sky(gaia_coords)
    #     match_radius = gaia_match_radius * u.arcsec
    #     ok_match = (d2d < match_radius)

    #     gaia_gmag = np.full(len(ids), np.nan)
    #     gaia_sid = np.full(len(ids), -1, dtype=np.int64)
    #     gaia_sep = np.full(len(ids), np.nan)
    #     gaia_ra_matched = np.full(len(ids), np.nan)
    #     gaia_dec_matched = np.full(len(ids), np.nan)

    #     gaia_gmag[ok_match] = gaia_tab["phot_g_mean_mag"][idx[ok_match]]
    #     gaia_sid[ok_match] = gaia_tab["source_id"][idx[ok_match]]
    #     gaia_sep[ok_match] = d2d[ok_match].to_value(u.arcsec)
    #     gaia_ra_matched[ok_match] = gaia_tab["ra"][idx[ok_match]]
    #     gaia_dec_matched[ok_match] = gaia_tab["dec"][idx[ok_match]]

    #     print(f"Gaia matches: {int(np.sum(ok_match))}/{len(ids)}")

    #     # --- D. BUILD UPDATED META TABLE ---
    #     meta_tab = Table(
    #         rows=[
    #             (int(i), float(star_positions_0[i, 0]), float(star_positions_0[i, 1]),
    #              float(ra_deg[i]), float(dec_deg[i]),
    #              float(gaia_gmag[i]), int(gaia_sid[i]), float(gaia_sep[i]),
    #              float(gaia_ra_matched[i]), float(gaia_dec_matched[i]))
    #             for i in ids
    #         ],
    #         names=["id", "x0", "y0", "RA_deg", "DEC_deg", "Gaia_Gmag", 
    #                "Gaia_source_id", "Gaia_sep_arcsec", "Gaia_RA", "Gaia_Dec"]
    #     )

    #     # Attach meta to tables
    #     coords_with_meta = attach_meta_to_table(coords_tab, meta_tab)
    #     mags_with_meta = attach_meta_to_table(mags_tab, meta_tab)
    #     magsdiff_with_meta = attach_meta_to_table(mags_diff, meta_tab)

    #     # Save tables
    #     coords_with_meta.write(coords_txt, format="ascii.tab", overwrite=True)
    #     mags_with_meta.write(mags_txt, format="ascii.tab", overwrite=True)
    #     magsdiff_with_meta.write(mags_with_ref_and_diff_txt, format="ascii.tab", overwrite=True)
    #     print("Re-wrote output tables with full Gaia metadata.")

    #     # --- E. RUN THE 3 DIAGNOSTICS ---
        
    #     # Diag 1: WCS Overlay
    #     diagnostic_1_visual_overlay(first_frame, wcs_header, gaia_tab, star_positions_0)
        
    #     # Diag 2: Mag Correlation
    #     diagnostic_2_mag_correlation(mags_with_meta)
        
    #     # Diag 3: Astrometric Residuals
    #     diagnostic_3_residuals(mags_with_meta)

    #     # --- F. FINAL SNR PLOT ---
    #     tab_for_snr = mags_with_meta if snr_flux_col.startswith("rel_flux_idl") else magsdiff_with_meta
    #     snr_vs_gaia_gmag(
    #         tab_for_snr,
    #         n_first_frames=snr_first_n,
    #         flux_col=snr_flux_col,
    #         gaia_mag_col="Gaia_Gmag",
    #         out_txt=snr_out_txt,
    #         out_png=snr_out_png,
    #         min_points=5
    #     )

    # except Exception as e:
    #     print("WARNING: Online astrometry/Gaia step failed.")
    #     print(f"  Reason: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     print("  Your photometry outputs WITHOUT RA/Dec/Gaia were still produced.")