from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import asyncio
import os
from pathlib import Path

import omni.kit.asset_converter
from isaacsim.core.utils.extensions import enable_extension

SIM_DIR = Path(__file__).resolve().parent
HOME_DIR = Path.home()
DOWNLOADS_DIR = Path(os.environ.get("ROBOT_CAPSTONE_DOWNLOADS_DIR", HOME_DIR / "Downloads")).expanduser()

INPUTS = {
    "Apple": DOWNLOADS_DIR / "Apple.glb",
    "Red_Ball": DOWNLOADS_DIR / "red-ball.glb",
    "Basket": DOWNLOADS_DIR / "minecart.glb",
    # poly.pizza/m/1L9oJAw6nY2 — "Phone" by Alex Safayan, CC-BY 3.0.
    # Chosen over the other phone GLB in Downloads because it is 11.7% thick
    # (vs 6.6%) and multi-coloured: from the overhead camera 2 m up a phone is
    # only ~50x25 px, and thickness-shadow plus a dark screen against a lighter
    # body are the only cues that survive that downsampling.
    "Phone": DOWNLOADS_DIR / "Phone by Alex Safayan - 1L9oJAw6nY2.glb",
}
OUTPUT_DIR = SIM_DIR / "assets" / "imported"


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

    # Only convert what is not already there. The USDs under assets/imported/
    # are TRACKED IN GIT, and re-converting an unchanged GLB rewrites the file
    # byte-for-byte differently, so an unconditional pass dirties the working
    # tree with four functionally identical assets every time someone adds one.
    # Delete the .usd to force a regeneration.
    todo = {
        name: source
        for name, source in INPUTS.items()
        if not (OUTPUT_DIR / f"{name}.usd").exists()
    }
    skipped = sorted(set(INPUTS) - set(todo))
    if skipped:
        print(f"Already converted, skipping: {', '.join(skipped)}")
    if not todo:
        print("Nothing to do.")
        return

    missing = [str(path) for path in todo.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input assets: {missing}")

    for name, source in todo.items():
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
