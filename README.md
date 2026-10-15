# Room inspection

Detects what has changed in the room, measures widths of objects, can tell real changes from things that were just hidden, displacement of objects based on a shared 3D model combining two video recordings.

See [DESIGN.md](DESIGN.md) for how it works, design choices and accuracy.

## The result

- Vase moved about 103 cm
- Handbag added where the vase used to be
- Paper towel roll removed

## Requirements

- Windows (run scripts are .bat files)
- COLMAP, CUDA build (expected at C:\colmap)
- ffmpeg (must be on the PATH)
- Python 3.12 (expected at C:\v) in a virtual environment install commands given below:

```
python -m venv C:\v
C:\v\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
C:\v\Scripts\pip install -r requirements.txt
```
The last command MUST be ran from inside the project folder

NOTE: Both paths for COLMAP and Python can be changed at the top of the .bat files

## How to run

1. Put VideoOne.MOV (before) and VideoTwo.MOV (after) in the same folder as the scripts.

2. Run colmap_steps.bat, pauses for one step done within the COLMAP GUI and takes ~1.5 hours for mostly dense reconstruction, instructions for the GUI step are written inside the file.

3. You need to run `run.bat`, set KNOWN_CM at the top of scale.py to your measured length in cm of a certain area measured in the room, a window will open for the scale step for only the first time, Shift+click two ends of something you have measured, then press Q. Measured length is set at the top of scale.py, length is set as KNOWN_CM. First run downloads the CLIP model (600 MB).

4. Results appear in output/report.txt

NOTE: Everything is created in the same folder as the scripts.

## Inputs and outputs

Inputs: 

- examples/input_frames/ - four frames from two videos, full videos not included for privacy which include location data

Outputs: 

- `report.txt` (Plain English result)
- `results.json` (All detections and verdict)
- `changes.json` (Verified changes)
- `panels/` (Before and after photos)
- `changes_3d.html` (Interactive 3D to download and open in a browser)
- `changes_3d.png` (Picture output)

Examples from my room outputs are in examples/output/ and the same files appear in output/ when the system is run.

## AI use

There has been use of AI for the purposes of coding help, such as suggesting the approach, majority of the code coverage and debugging any necessary errors which may have occurred within the process.

