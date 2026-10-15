@echo off
rem ---------------------------------------------------------------------------
rem Reconstruction steps, run once before run.bat. Needs ffmpeg and COLMAP (CUDA build).
rem Put the two phone videos in this folder as VideoOne.MOV (before) and VideoTwo.MOV (after).
rem Dense stereo takes roughly 45 minutes per video on a laptop RTX GPU.
rem ---------------------------------------------------------------------------
set COLMAP=C:\colmap\bin\colmap.exe
cd /d %~dp0

rem 1. Frames: 2 per second, high JPEG quality. a_ = video A (before), b_ = video B (after)
if not exist images mkdir images
ffmpeg -i VideoOne.MOV -vf fps=2 -qscale:v 2 images\a_%%04d.jpg
ffmpeg -i VideoTwo.MOV -vf fps=2 -qscale:v 2 images\b_%%04d.jpg

rem 2. ONE joint sparse reconstruction of both videos, so they share a coordinate frame.
rem    I ran this in the COLMAP GUI: Reconstruction > Automatic reconstruction,
rem    workspace = ws, images = images, Shared intrinsics ON, Dense model OFF.
echo Run step 2 in the COLMAP GUI now (see comments in this file), then press a key.
pause

rem 3. Export the model as text (read by the Python scripts)
%COLMAP% model_converter --input_path ws\sparse\0 --output_path ws\sparse\0 --output_type TXT

rem 4. Split the joint model into an A-only and a B-only model (same coordinate frame)
cd ws
if not exist sparse_a mkdir sparse_a
if not exist sparse_b mkdir sparse_b
dir /b ..\images\a_*.jpg > a_names.txt
dir /b ..\images\b_*.jpg > b_names.txt
%COLMAP% image_deleter --input_path sparse\0 --output_path sparse_a --image_names_path b_names.txt
%COLMAP% image_deleter --input_path sparse\0 --output_path sparse_b --image_names_path a_names.txt

rem 5. Dense cloud for each video
for %%v in (a b) do (
  %COLMAP% image_undistorter --image_path ..\images --input_path sparse_%%v --output_path dense_%%v --output_type COLMAP --max_image_size 1600
  %COLMAP% patch_match_stereo --workspace_path dense_%%v --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
  %COLMAP% stereo_fusion --workspace_path dense_%%v --workspace_format COLMAP --input_type geometric --output_path dense_%%v\fused.ply
)
cd ..

rem 6. Copy the two dense clouds to where detect.py expects them
if not exist clouds mkdir clouds
copy /Y ws\dense_a\fused.ply clouds\a.ply
copy /Y ws\dense_b\fused.ply clouds\b.ply
echo Reconstruction done. Next: run.bat
