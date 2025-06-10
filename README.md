# Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins
[Project Page](https://prompting-with-the-future.github.io/) 

<img  src="intro.gif" width="800">

Official implementation of Prompting with the Future (RSS 2025). We provide a demo on a scanned environment and also provide a pipeline for scanning your own environment for open-world manipulation.

## Installation
```
git clone https://github.com/prompting-with-the-future/prompting-with-the-future.git
cd prompting-with-the-future
conda env create --file environment.yaml
conda activate pwf
pip install --upgrade mani_skill
conda install pytorch3d -c pytorch3d
```

Please follow the instructions in the [2d Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) and repo to install the dependency for reconstruction.

Since the environment for SAM2[https://github.com/facebookresearch/sam2] is not compatible with the main environment and it is only used for reconstruction, we provide a separate environment for SAM2.

```
conda create -n sam2 python=3.10.0
conda activate sam2
cd sam2
pip install -e .
```

## Run our demo
We prepared one scaned environment for testing the Prompting with the Future.

``` 
python main.py --scene_name basket_world --instruction "put the green cucumber into the basket"
```

The resulting trajectory and joint actions will be saved in the `results` folder.

## Scan your own environment
We provide two ways to scan your own environment for open-world manipulation.

### 1. Scan with an phone (recommended)
Firstly, print a checkerboard (utils/reconstruct/checker_board.svg) and put it in the workspace.
Then, use a phone camera to flexibly scan your environment. (60fps is recommended)
Name your scene as `{SCENE_NAME}`.
Put the video in the `gaussians/colmap/{SCENE_NAME}` folder.
Run the following script to reconstruct the interactive digital twin with movable meshes and gaussians.

```
sh build.sh {SCENE_NAME} {INSTRUCTION}
```

### 2. Scan with a robot
We also provide a script to use the robot to scan your environment.
Due to different robot platforms, we provide an example script on Droid[https://droid-dataset.github.io/] setup for scanning with a robot.
You can adapt the `utils/reconstruct/robot_scan.py` script for your own robot.

The planning code is the same as the previous one.

## Post-processing (optional)
We found that VLMs are quite robust to the artifacts in the reconstructed scene.
However, we still provide a post-processing step to improve the visual quality.

Change the `box` parameters in the `gaussians/gaussian_world.py` script to the bounding box of your workspace. This will remove the gaussians outside the bounding box.

The post-processing flag will also fill the holes under the objects.

## Planning
Start the planning on your own environment by running the following command.
```
python main.py --scene_name {SCENE_NAME} --instruction {INSTRUCTION} --name {EXP_NAME}
```