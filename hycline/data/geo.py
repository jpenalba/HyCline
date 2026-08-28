"""Coordinate-based transect distance calculation.

HZAR itself expects a pre-computed 1D `distance` per site -- its own bundled
`manakinLocations` dataset documents itself simply as "distance from
locality A for each locality," i.e. a great-circle distance from one
reference site, computed by hand before the data ever reaches HZAR. Most
users start from GPS coordinates, not a pre-computed transect distance, so
this module fills that gap.

This is explicitly NOT the 2D spatial cline surface extension described in
the project architecture plan (fitting a cline as a smooth function over 2D
space -- a genuinely harder, still-unimplemented research problem). This
module does something much simpler and safer: collapse 2D coordinates down
to a 1D distance so the existing 1D cline model can be used, the same way a
field biologist manually measuring distance along a transect tape would.
Whether that collapse is a reasonable approximation depends on the
transect's actual shape -- see `project_onto_transect_axis`'s docstring for
when to prefer it over the simpler `distance_from_reference`.
"""

from __future__ import annotations

import numpy as np

from hycline.data.schema import values_consistent

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between point(s) (lat1, lon1) and
    (lat2, lon2). Inputs may be scalars or arrays (broadcast normally)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                               for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_from_reference(lat, lon, ref_lat: float | None = None,
                             ref_lon: float | None = None) -> np.ndarray:
    """Great-circle distance (km) from each site to a single reference
    point. If no reference is given, defaults to the first site in the
    input arrays -- this matches HZAR's own convention (its bundled
    `manakinLocations` dataset is literally "distance from locality A").

    Reasonable when the transect samples roughly radiate outward in one
    direction from a natural reference point (e.g. one edge of the hybrid
    zone). If sampling zigzags or the reference point isn't a natural
    transect endpoint, `project_onto_transect_axis` is usually a better fit.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = lat[0], lon[0]
    return haversine_distance(ref_lat, ref_lon, lat, lon)


def check_site_coordinate_consistency(site_labels, lat, lon) -> None:
    """Raise a clear `ValueError` naming the site if any two individuals
    sharing a site have different (lat, lon) values.

    Mirrors the existing "every individual at a site must share the
    same distance value" check the hard-count writers already enforce
    (`hycline.data.vcf.write_geographic_hard_count_csv`,
    `hycline.data.beagle.write_beagle_geographic_hard_count_csv`) --
    checking coordinates directly, before any distance is computed from
    them, gives a clearer error for a user who provided lat/lon rather
    than a pre-computed distance: pointing at the actual coordinates
    they supplied, not a "distance" value they never directly specified
    (which two slightly different coordinates would also trigger,
    downstream, once converted -- but with a far more confusing
    message).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    site_coord: dict = {}
    for site, la, lo in zip(site_labels, lat, lon):
        if site not in site_coord:
            site_coord[site] = (la, lo)
        elif not (values_consistent(site_coord[site][0], la)
                  and values_consistent(site_coord[site][1], lo)):
            prev_la, prev_lo = site_coord[site]
            raise ValueError(
                f"Site '{site}' has inconsistent coordinates across its individuals in "
                f"the metadata (({prev_la}, {prev_lo}) vs ({la}, {lo})) -- every "
                f"individual at the same site must share the same latitude/longitude."
            )


def project_onto_transect_axis(lat, lon) -> np.ndarray:
    """Collapse 2D coordinates onto their principal axis via PCA, returning
    a 1D distance (km) along that axis relative to the lowest-projected
    site.

    Coordinates are first converted to a local planar (equirectangular)
    approximation using the sample's mean latitude -- adequate at the
    sub-continental scales hybrid-zone transects are typically sampled at;
    not appropriate for sampling that spans a large fraction of the globe
    or crosses the antimeridian.

    More robust than `distance_from_reference` when the sampling isn't a
    straight line radiating from a natural endpoint (e.g. sites scattered
    somewhat off the main transect line, or no obvious reference site) --
    PCA finds the best-fit line through all the points and projects onto
    that, rather than anchoring to whichever site happens to be listed
    first. The tradeoff: the resulting distances depend on the full set of
    sampled points, so adding or removing sites can shift every other
    site's computed distance slightly, which `distance_from_reference`
    does not do.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    mean_lat_rad = np.radians(np.mean(lat))
    x = EARTH_RADIUS_KM * np.radians(lon) * np.cos(mean_lat_rad)
    y = EARTH_RADIUS_KM * np.radians(lat)
    coords = np.column_stack([x, y])
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projected = centered @ axis
    return projected - projected.min()
