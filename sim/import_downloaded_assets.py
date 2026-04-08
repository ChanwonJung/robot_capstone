from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import asyncio
import os
from pathlib import Path

import omni.kit.asset_converter
from isaacsim.core.utils.extensions import enable_extension

REPO_ROOT = Path(__file__).resolve().parent
HOME_DIR = Path.home()
DOWNLOADS_DIR = Path(os.environ.get("ROBOT_CAPSTONE_DOWNLOADS_DIR", HOME_DIR / "Downloads")).expanduser()
ISAACSIM_ROOT = REPO_ROOT / "isaacsim"

INPUTS = {
    "Water_Bottle": DOWNLOADS_DIR / "Water Bottle.glb",
    "Glass": DOWNLOADS_DIR / "ikea_glass.glb",
    "Apple": DOWNLOADS_DIR / "Apple.glb",
    "Red_Ball": DOWNLOADS_DIR / "red-ball.glb",
    "Blue_Cube": DOWNLOADS_DIR / "blue_cube.glb",
    "Book": DOWNLOADS_DIR / "book.glb",
}
OUTPUT_DIR = REPO_ROOT / "assets" / "imported"


async def convert(in_file: Path, out_file: Path):
    def progress_callback(progress, total_steps):
        pass

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.single_mesh = False
    context.smooth_normals = True
    context.use_meter_as_world_unit = True

    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(in_file), str(out_file), progress_callback, context
    )
    while True:
        success = await task.wait_until_finished()
        if success:
            return True
        await asyncio.sleep(0.1)


async def main():
    enable_extension("omni.kit.asset_converter")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input assets: {missing}")

    for name, source in INPUTS.items():
        target = OUTPUT_DIR / f"{name}.usd"
        print(f"Converting {source} -> {target}")
        ok = await convert(source, target)
        if not ok:
            raise RuntimeError(f"Failed to convert {source}")
        print(f"Saved: {target}")


if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(main())
    finally:
        simulation_app.close()
