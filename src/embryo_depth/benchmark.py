"""Measure what the viewer actually costs, at the operation the tool exists for.

The target is scrubbing ``t`` at a fixed depth. Shell size varies with depth -- it is
largest at the surface and shrinks toward the core -- so the worst case is small ``d``,
which is exactly where the work happens. Benchmarking at a mid-range depth would flatter
the result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .store import DEFAULT_STORE, Dataset
from .viewer import ShellSource, naive_shell_volume


def run(store: Path, levels: list[str], depths: list[int], half_width: int, n_t: int) -> None:
    ds = Dataset(store)
    for level in levels:
        lv = ds.levels[level]
        src = ShellSource(ds, level, range(min(n_t, ds.n_timepoints)))
        t0 = time.time()
        src.load(verbose=False)
        build = time.time() - t0
        resident = sum(i.nbytes for i in src.index.values()) / 2**30
        voxels = np.prod(lv.spatial_shape)
        print(
            f"\nlevel {level}  {lv.spatial_shape}  {voxels / 1e6:.1f}M voxels/timepoint\n"
            f"  index: {build:.1f}s for {len(src.timepoints)} timepoints, "
            f"{resident:.2f} GiB resident"
        )
        print(f"  {'depth':>6} {'shell vox':>11} {'points/frame':>13} {'volume/frame':>13}")
        for d in depths:
            counts, t_pts, t_vol = [], [], []
            for t in src.timepoints[: min(30, len(src.timepoints))]:
                s = time.time()
                coords, _ = src.shell(t, d, half_width)
                t_pts.append(time.time() - s)
                counts.append(len(coords))
                s = time.time()
                src.shell_volume(t, d, half_width)
                t_vol.append(time.time() - s)
            pts_ms = np.median(t_pts) * 1e3
            vol_ms = np.median(t_vol) * 1e3
            print(
                f"  {d:5d}um {np.median(counts) / 1e3:10.0f}k "
                f"{pts_ms:10.2f}ms {vol_ms:10.2f}ms"
                f"   -> {1000 / max(pts_ms, 1e-3):5.0f} / {1000 / max(vol_ms, 1e-3):4.0f} fps"
            )

        # correctness spot-check against the reference implementation
        t = src.timepoints[len(src.timepoints) // 2]
        image = np.asarray(ds.image(level)[t, 0])
        depth_units = np.asarray(ds.depth(level)[t, 0])
        ok = all(
            np.array_equal(
                src.shell_volume(t, d, half_width),
                naive_shell_volume(image, depth_units, d, half_width),
            )
            for d in depths
        )
        print(f"  index matches the naive full-volume computation at t={t}: {ok}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--levels", nargs="+", default=["3", "2"])
    p.add_argument("--depths", nargs="+", type=int, default=[2, 10, 30, 60])
    p.add_argument("--half-width", type=int, default=3)
    p.add_argument("--timepoints", type=int, default=40)
    args = p.parse_args()
    run(args.store, args.levels, args.depths, args.half_width, args.timepoints)


if __name__ == "__main__":
    main()
