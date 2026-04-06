from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import asyncio
from pathlib import Path

import omni.kit.asset_converter
from isaacsim.core.utils.extensions import enable_extension


INPUTS = {
    "Phone": Path("/home/chanwonjung/Downloads/Phone.glb"),
    "Water_Bottle": Path("/home/chanwonjung/Downloads/Water Bottle.glb"),
    "Coffee_Cup": Path("/home/chanwonjung/Downloads/Coffee cup.glb"),
    "Glass": Path("/home/chanwonjung/Downloads/ikea_glass.glb"),
    "Apple": Path("/home/chanwonjung/Downloads/Apple.glb"),
    "Red_Ball": Path("/home/chanwonjung/Downloads/red-ball.glb"),
    "Blue_Cube": Path("/home/chanwonjung/Downloads/blue_cube.glb"),
    "Book": Path("/home/chanwonjung/Downloads/book.glb"),
}
OUTPUT_DIR = Path("/home/chanwonjung/robot_capstone/assets/imported")


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
