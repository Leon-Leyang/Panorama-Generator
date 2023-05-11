# Panorama-Generator
This repository contains the source code for the coursework of COMP3065 Computer Vision. The project focuses on developing a panorama generator that takes a video as input and produces a panoramic image as the output.

### Data Preparation

To generate panoramas, you have two options for the input data:

1. **Use Your Own Video:** You can input your own video footage to create a panorama. Ensure that the video captures a scene from multiple perspectives, allowing for the creation of a wide-angle composite image.

2. **Sample Data:** Alternatively, you can use the provided sample data, which can be found [here](https://drive.google.com/file/d/1ySHbzh1f_BRxbjbvKUDrekkmOEWhvgl7/view?usp=sharing). The sample data consists of 6 clips of videos captured with a tripod, resulting in stable footage. Additionally, there are 3 clips captured while handheld, introducing some challenges due to camera movement. The sample data covers a total of 6 different scenarios.

### Installation

To run the panorama generator, you will need several Python packages. You can easily install these packages by running the following command:

```bash
pip install -r requirements.txt
```

This command will install all the required packages specified in the `requirements.txt` file. Make sure you have Python and pip installed on your system before running this command.

### Usage

To run the panorama generator, use the following command:

```bash
python main.py -v <video_path> -i <interval> -w <width> -he <height> -r <ref_frame_idx> -l <num_levels> -m <mask_type> -c -o <output_path>
```

An example command is:

```bash
python main.py -v ./data/stable/garden_stable.mp4 -i 24 -c -o output_panorama.jpg
```

Here is an explanation of the parameters:

- `-v` or `--video`: Specifies the path to the video file.
- `-i` or `--interval`: Sets the interval between sampled frames (default is 72).
- `-w` or `--width`: Specifies the width of the sampled frames (default is 1920).
- `-he` or `--height`: Specifies the height of the sampled frames (default is 1080).
- `-r` or `--ref_frame_idx`: Sets the index of the reference frame (default is 0). The reference frame determines the coordinate system for the entire panorama.
- `-l` or `--num_levels`: Specifies the number of levels in the pyramid of multi-band blending (default is 3).
- `-m` or `--mask_type`: Sets the type of mask used in multi-band blending (default is 'feathered'). Other available options include 'direct' and 'gaussian'.
- `-c` or `--crop`: When this flag is included in the command, it enables automatic cropping of the surrounding black borders around the generated panorama.
- `-o` or `--output`: Specifies the path to the output panorama (default is 'output.jpg').

## Repository Organization

The directory structure of this repository is organized as follows:

```
├── main.py
├── .gitignore
├── requirements.txt
├── README.md
└── utils
    ├── __init__.py
    └── utils.py
    └── panorama_generator.py

```

- `main.py`: The main entry point of the panorama generator program.
- `.gitignore`: Specifies files and directories to be ignored by version control.
- `requirements.txt`: Lists the required Python packages and their versions.
- `README.md`: The README file you are currently reading.

The `utils` directory contains utility files and the main panorama generator class:

- `utils.py`: Contains utility functions used by the panorama generator.
- `panorama_generator.py`: Contains the main class for the panorama generator.
