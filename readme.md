# droid_metric

This repo is for project combind [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [Metric3D](https://github.com/YvanYin/Metric3D), taking metric depth to improve the performance of DROID-SLAM in monocular mode.

### installation
```bash
# clone the repo with '--recursive' to get the submodules
# or run 'git submodule update --init --recursive' after cloning
cd droid_metric

# create conda env
conda create -n droid_metric python=3.9
conda activate droid_metric

# install pytorch (other versions may also work)
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# install python packages
pip install -r requirements.txt

# intsall droid-slam-backend
cd module/droid_slam
python setup.py install
cd ../..
```

### usage
###### 1. pretrained models
Download the pretrained model following the official page of [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [Metric3D](https://github.com/YvanYin/Metric3D).


###### 2. utils
For camera calibration, check `scripts/calib.py`
For video sampling, check `scripts/sample.py`

###### 3. run
```bash
## depth estimate
python depth.py --images $/path/to/images --out $/path/to/output --intr $/path/to/intrinsic/file
# for more options, check `depth.py`

## droid-slam
python run.py --rgb $/path/to/rgb/dir --depth $/path/to/depth/dir --intr $/path/to/intrinsic/file --viz
# for more options, check `run.py`

## mesh recon
python mesh.py --rgb $/path/to/rgb/dir --depth $/path/to/depth/dir --traj $/path/to/pose/dir --intr $/path/to/intrinsic/file --mesh $/path/to/output/mesh/ply
# for more options, check `mesh.py`
```

##### 4.scripts
```bash
## test depth estimate, droid slam and mesh reconstruction for rgb image sequence
python -m scripts.test_seq --rgb $/path/to/rgb/dir --depth $/path/to/depth/dir --poses $/path/to/pose/dir --mesh $/path/to/output/mesh/ply --intr $/path/to/intrinsic/file --viz
```


### preview
![without depth](assets/w_o_depth.png)

***w/o metric depth***

![with depth](assets/w_depth.png)

***w/ metric depth***

