# Generalizable Image Repair for Robust Visual Autonomous Racing

### Paper Link: [arXiv](https://arxiv.org/abs/2503.05911)

### Configuring the Simulator

A modified version of the DonkeyCar simulator is required to run our experiments. Our modified simulator can be found [here](https://github.com/Trustworthy-Engineered-Autonomy-Lab/donkey-unity-sim).

We have modified the simulator to:
- Add pausing functionality
- Add options to change lighting
- Add CTE display and fix CTE calculation after car resets

Unity is required to compile the simulator into an executable.

### Generating Training Data

For convenience, we have provided training data for the Mini Monaco track [here](), but you can also use the simulator to generate your own training data. To generate training data:
- Install our modified version of the `donkeycar` package using `pip install ./donkeycar`.
- Run the simualtor by running `python manage.py drive`.
    - You will need to tweak the `DONKEY_SIM_PATH` variable in `myconfig.py` file to have the correct simulator path.
- Go to the browser interface at `http://localhost:8887`.
- Click **Start Recording**.
- Manually drive the car around the track.

The images and corresponding control actions (tubs) will then be saved in ```data/```. Then, the `ImageAugmentor` class in `noise_generator.py` can be used to augment the data with various disturbances, such as darkness, rain, fog, snow, and salt/pepper noise.

### Training a Controller

To train a controller model using paired images and control actions, run:
```
python train.py --model=<model path>.h5 --tubs=<data path>
```
Where `<data path>` is the path to the tub folder you wish to use that contains a manifest.json (for example `data/Mini_Monaco`) and `<model path>` is the desired location to save the trained model.

### Preparing CycleGAN and pix2pix

CycleGAN and pix2pix training files are located in `pytorch-CycleGAN-and-pix2pix/`.

First, use the `convert.ipynb` file to convert the .h5 controller model to an onnx model for use in the controller loss calculation.

Then, create the image dataset. To do so, create a folder in the `datasets/` folder with the following structure:
- `datasets/`
    - `trainA/`
    - `trainB/`
    - `testA/`

where `trainA` is the set of corrupted images, `trainB` contains paired uncorrupted images with the same file names as those in `trainA`, and `testA` contains test corrupted images.

To make an aligned (paired) dataset, duplicate `testA` to `testB`. Then, run the provided `make_dataset_aligned.py` file. If all goes well, `train` and `test` folders will be created consisting of paired observations.

If you wish to use the controller loss, you must also create a `controls/` folder alongside the others, which contains the corresponding `.catalog` files that contain the control actions. The final directory structure should look as follows:
- `datasets/`
    - `controls/`
        - `*.catalog`
    - `train/`
        - `*.png`
    - `trainA/`
        - `*.png`   
    - `trainB/`
        - `*.png`
    - `test/`
        - `*.png`
    - `testA/`
        - `*.png`
    - `testB/`
        - `*.png`

### Training CycleGAN

To train a CycleGAN model, run the following command:
```
python train.py --dataroot <dataset path> --name <name> --controller_path <controller path> --model <cycle_gan|cycle_gan_controller> --dataset_mode <unaligned|control> --preprocess none --batch_size 16 --checkpoints_dir <checkpoint path>
```
where:
- `<dataset path>` is the path to the previously created dataset.
- `<name>` is what you want to name the current training run.
- `<controller_path>` is the path to the onnx controller.
- `model` is the model type (either plain CycleGAN or with controller loss).
- `dataset_mode` is the dataset mode (should correspond to the previous two model types).
- `<checkpoints_dir>` is the directory you want to save the model training checkpoints.

Out of the saved models, the one to use for restoring images is `*_net_G_A.pth`.

### Training pix2pix

To train a pix2pix model, run the following command:
```
python train.py --dataroot <dataset path> --name <name> --controller_path <controller path> --model <pix2pix|pix2pix_controller> --dataset_mode <aligned|control> --preprocess none --netG resnet_9blocks --batch_size 16 --checkpoints_dir <checkpoint path>
```
where:
- `<dataset path>` is the path to the previously created dataset.
- `<name>` is what you want to name the current training run.
- `<controller_path>` is the pssath to the onnx controller (only required if using the controller loss).
- `model` is the model type (either plain pix2pix or with controller loss).
- `dataset_mode` is the dataset mode (should correspond to the previous two model types).
- `<checkpoints_dir>` is the directory you want to save the model training checkpoints.

Out of the saved models, the one to use for restoring images is `*_net_G.pth`.

### Testing Repair Models

To run the simulator with different repair models across various noises and log CTE values, we use the `manage.py` script as follows:
```
python manage.py drive --model=<controller path> --meta=<cyclegan|pix2pix>:<GAN path> --meta=noise:<normal|brightness|rain|fog|snow|salt_pepper> --meta=name:<name>
```
where:
- `<controller path>` is the path to the trained .h5 controller.
- `<cyclegan|pix2pix>` and `<GAN path>` designate the type and location of the provided repair model.
- `noise` designates the noise applied to images before they are passed to the repair models and controller.
- `name` appends a name to the results folder to help identify the current run.

The simulator will run for 1500 iterations (about 1 full lap around Mini Monaco), and then close itself. The number of iterations can be tweaked in `myconfig.py`. During operation, corrupted images and their repaired counterparts are stored in the results folder, alongside a `.csv` file with CTE values.

### Acknowledgement

This codebase is built on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [sdsandbox](https://github.com/tawnkramer/sdsandbox).