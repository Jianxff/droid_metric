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
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# install python packages
pip install -r requirements.txt
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
python -m scripts.predict --images $/path/to/images --out $/path/to/output
# for more options, check `scripts/predict.py`

## droid-slam
python -m scripts.run --rgb $/path/to/rgb/dir --depth $/path/to/depth/dir --viz
# for more options, check `scripts/run.py`
```


### preview
![without depth](assets/w_o_depth.png)
***w/o metric depth***

![with depth](assets/w_depth.png)
***w/ metric depth***

