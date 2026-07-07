#!/usr/bin/env python3
"""
graspgen_cli.py — Standalone ZMQ client for GraspGen inference server.

Usage examples:
  # Synthetic cube (1000 points)
  python3 graspgen_cli.py --host 127.0.0.1 --port 5556 --shape cube --num-grasps 100 --topk 10

  # Synthetic sphere (800 points)
  python3 graspgen_cli.py --host 127.0.0.1 --port 5556 --shape sphere --num-grasps 100 --topk 10

  # From file (JSON, NPY, or NPZ)
  python3 graspgen_cli.py --host 127.0.0.1 --port 5556 --from-file my_cloud.npy --num-grasps 100 --topk 10

Prerequisites:
  pip install pyzmq msgpack msgpack-numpy numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from zmq_client import GraspGenClient


def gen_cube(num_points: int = 1000, size: float = 0.1, center: tuple = (0, 0, 0.1)) -> np.ndarray:
    """Generate synthetic cube point cloud.

    Parameters
    ----------
    num_points : int
        Number of points to sample
    size : float
        Cube side length (m)
    center : tuple
        (x, y, z) center in world frame

    Returns
    -------
    cloud : (N, 3) float32
        Point cloud in world frame
    """
    half = size / 2
    cx, cy, cz = center

    # Sample uniformly from cube volume
    pts = np.random.uniform(-half, half, size=(num_points, 3)).astype(np.float32)
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[:, 2] += cz
    return pts


def gen_sphere(num_points: int = 1000, radius: float = 0.05, center: tuple = (0, 0, 0.1)) -> np.ndarray:
    """Generate synthetic sphere point cloud.

    Parameters
    ----------
    num_points : int
        Number of points to sample
    radius : float
        Sphere radius (m)
    center : tuple
        (x, y, z) center in world frame

    Returns
    -------
    cloud : (N, 3) float32
        Point cloud in world frame
    """
    # Random points on sphere surface + some interior
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    # Add some interior points
    interior = np.random.uniform(0, radius, size=(num_points // 3, 1))
    r_interior = interior * np.random.uniform(0, 1, size=(num_points // 3, 1)) ** (1/3)

    idx = np.random.choice(num_points, size=num_points // 3, replace=False)
    x[idx] = r_interior[:, 0] * np.cos(theta[idx]) * np.sin(phi[idx])
    y[idx] = r_interior[:, 0] * np.sin(theta[idx]) * np.sin(phi[idx])
    z[idx] = r_interior[:, 0] * np.cos(phi[idx])

    cloud = np.column_stack([x, y, z]).astype(np.float32)
    cloud[:, 0] += center[0]
    cloud[:, 1] += center[1]
    cloud[:, 2] += center[2]
    return cloud


def load_cloud(path: str) -> np.ndarray:
    """Load point cloud from file (JSON, NPY, NPZ).

    Parameters
    ----------
    path : str
        Path to cloud file

    Returns
    -------
    cloud : (N, 3) float32
    """
    p = Path(path)

    if p.suffix == '.json':
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and 'points' in data:
            cloud = np.array(data['points'], dtype=np.float32)
        else:
            cloud = np.array(data, dtype=np.float32)
    elif p.suffix == '.npy':
        cloud = np.load(p).astype(np.float32)
    elif p.suffix == '.npz':
        npz = np.load(p)
        # Try common keys
        for key in ['cloud', 'points', 'point_cloud', 'arr_0']:
            if key in npz:
                cloud = npz[key].astype(np.float32)
                break
        else:
            raise ValueError(f"NPZ has no recognized key. Available: {list(npz.keys())}")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"Cloud must be (N, 3), got {cloud.shape}")

    return cloud


def format_output(grasps: np.ndarray, confidences: np.ndarray, num_show: int = None) -> str:
    """Format grasp output for CLI display."""
    M = len(confidences)
    if num_show is None:
        num_show = min(5, M)

    lines = [
        f"\n{'='*60}",
        f"GraspGen Inference Result",
        f"{'='*60}",
        f"Total grasps returned: {M}",
        f"\nTop {num_show} grasps by confidence:\n",
    ]

    for i in range(num_show):
        conf = confidences[i]
        grasp = grasps[i]
        # Extract position (last column of 4x4 matrix)
        pos = grasp[:3, 3]
        # Extract rotation angles (for display)
        lines.append(f"  [{i+1}] conf={conf:.4f}")
        lines.append(f"      pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")
        lines.append(f"      rotation=\n{grasp[:3, :3]}")
        lines.append("")

    if num_show < M:
        lines.append(f"... and {M - num_show} more grasps")

    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='GraspGen ZMQ CLI — test synthetic clouds or load your own',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic cube
  %(prog)s --host 127.0.0.1 --port 5556 --shape cube --num-grasps 100 --topk 10

  # From your saved point cloud
  %(prog)s --host 127.0.0.1 --port 5556 --from-file my_cloud.npy --num-grasps 100 --topk 10

  # Synthetic sphere with custom center
  %(prog)s --host 127.0.0.1 --port 5556 --shape sphere --center 0.1 0.0 0.05 --num-grasps 50 --topk 5
        """
    )

    # Server connection
    parser.add_argument('--host', default='127.0.0.1', help='GraspGen server host (default: %(default)s)')
    parser.add_argument('--port', type=int, default=5556, help='GraspGen server port (default: %(default)s)')
    parser.add_argument('--timeout-ms', type=int, default=10000, help='ZMQ timeout in ms (default: %(default)s)')

    # Cloud input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shape', choices=['cube', 'sphere'], help='Generate synthetic cloud')
    group.add_argument('--from-file', type=str, help='Load cloud from JSON/NPY/NPZ file')

    # Synthetic options
    parser.add_argument('--num-points', type=int, default=1000, help='Points in synthetic cloud (default: %(default)s)')
    parser.add_argument('--size', type=float, default=0.1, help='Cube size or sphere radius (default: %(default)s)')
    parser.add_argument('--center', type=float, nargs=3, default=[0, 0, 0.1], help='Cloud center x y z (default: %(default)s)')

    # Inference options
    parser.add_argument('--num-grasps', type=int, default=100, help='Total grasps to generate (default: %(default)s)')
    parser.add_argument('--topk', type=int, default=10, help='Top-K grasps to return (default: %(default)s)')
    parser.add_argument('--show', type=int, default=5, help='Number of grasps to display (default: %(default)s)')

    # Output
    parser.add_argument('--save-grasps', type=str, help='Save grasps as NPZ file')
    parser.add_argument('--save-cloud', type=str, help='Save input cloud as NPY file')

    args = parser.parse_args()

    # Load or generate cloud
    print(f"\n[*] Preparing point cloud...", file=sys.stderr)
    if args.from_file:
        cloud = load_cloud(args.from_file)
        print(f"    Loaded from {args.from_file}: {cloud.shape[0]} points", file=sys.stderr)
    else:
        if args.shape == 'cube':
            cloud = gen_cube(args.num_points, args.size, tuple(args.center))
        else:  # sphere
            cloud = gen_sphere(args.num_points, args.size, tuple(args.center))
        print(f"    Generated {args.shape}: {cloud.shape[0]} points, size={args.size} m", file=sys.stderr)

    if args.save_cloud:
        np.save(args.save_cloud, cloud)
        print(f"    Saved cloud to {args.save_cloud}", file=sys.stderr)

    # Connect to server
    print(f"\n[*] Connecting to GraspGen server at {args.host}:{args.port}...", file=sys.stderr)
    try:
        with GraspGenClient(args.host, args.port, args.timeout_ms) as client:
            print(f"    Connected.", file=sys.stderr)

            print(f"\n[*] Sending inference request...", file=sys.stderr)
            print(f"    num_grasps={args.num_grasps}, topk={args.topk}", file=sys.stderr)
            grasps, confidences = client.request(cloud, args.num_grasps, args.topk)
            print(f"    Received {len(confidences)} grasp candidates", file=sys.stderr)
    except RuntimeError as e:
        print(f"\n[!] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save results
    if args.save_grasps:
        np.savez(args.save_grasps, grasps=grasps, confidences=confidences)
        print(f"[*] Saved grasps to {args.save_grasps}", file=sys.stderr)

    # Display results to stdout
    print(format_output(grasps, confidences, args.show))


if __name__ == '__main__':
    main()
