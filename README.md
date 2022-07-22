# Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives
The experimental code of "[Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives](https://arxiv.org/abs/2207.10395)" in ECCV 2022.

## Abstract
Recently, Implicit Neural Representations (INRs) parameterized by neural networks have emerged as a powerful and promising tool to represent different kinds of signals due to its continuous, differentiable properties, showing superiorities to classical discretized representations. However, the training of neural networks for INRs only utilizes input-output pairs, and the derivatives of the target output with respect to the input, which can be accessed in some cases, are usually ignored. In this paper, we propose a training paradigm for INRs whose target output is image pixels, to encode image derivatives in addition to image values in the neural network. Specifically, we use finite differences to approximate image derivatives. We show how the training paradigm can be leveraged to solve typical INRs problems, i.e., image regression and inverse rendering, and demonstrate this training paradigm can improve the data-efficiency and generalization capabilities of INRs.

<img src='imgs/pipeline.png'/>

## Setup

### Environment

* Clone this repo
    ```shell
    git clone https://github.com/megvii-research/Sobolev_INRs.git
    cd Sobolev_INRs
    ```
* Install dependencies
    <details>
        <summary> Python 3 dependencies (click to expand) </summary>

    * PyTorch >= 1.10
    * torchvision
    * ConfigArgParse
    * einops
    * imageio
    * kornia
    * matplotlib
    * numpy
    * opencv_python
    * Pillow
    * scipy
    * tqdm
    </details>

    To setup a conda environment:
    ```shell
    conda create -n st_inrs python=3.7
    conda activate st_inrs
    pip install -r requirements.txt
    ```
### Data
* Create a directory with command:
    ```shell
    mkdir data
    ```
* Download data:
    * Download [Set5](https://drive.google.com/file/d/1C-C2eZIO3AQYi48EJ92MNWmWBGMdRYB6/view?usp=sharing) for __image regression__ task.
    * Download LLFF data from [NeRF authors' drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) for __inverse rendering__ task.
    * Download `gt_bach.wav` and `gt_counting.wav` from [SIREN authors' drive](https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K) for __audio regression__ task. Put two WAV files to folder `Audio`.
* Create soft links:
    ```shell
    ln -s [path to nerf_llff_data] ./data
    ln -s [path to Set5] ./data
    ln -s [path to Audio] ./data
    ```

## Reproducing Experiments
### Image Regression
```shell
cd Experiments/image_regression
python main.py --config [config txt file]
```
For example, training with __value and derivative supervision__ on *Baby* with a __sine-based__ model:
```shell
python main.py --config configs/baby/val_der/sine.txt
```
### Inverse Rendering
```shell
cd Experiments/inverse_rendering
python train.py --config [config txt file]
```
For example, train with __value and derivative supervision__ on *Fern* with a __ReLU-based__ MLP:
```shell
python train.py --config configs/fern/val_der/relu.txt
```
After training for 400K iterations, you can find novel view results in `logs/fern/val_der/relu/testset_400000`, you can evaluate results with following command and `score.txt` will be generated in `logs/fern/val_der/relu/testset_400000/score.txt`:
```shell
python eval.py --config configs/fern/val_der/relu.txt
```
### Audio Regression
```shell
cd Experiments/audio_regression
python main.py --config [config txt file]
```
For example, training with __value supervision__ and on *Bach* with a __sine-based__ model:
```shell
python main.py --config configs/bach/val_sine.txt
```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{yuan2022sobolev,
  title={Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives},
  author={Wentao Yuan and Qingtian Zhu and Xiangyue Liu and Yikang Ding and Haotian Zhang and Chi Zhang},
  year={2022},
  booktitle={ECCV},
```
## Acknowledgements
Some codes of image regression task and audio regression task are borrowed from [SIREN](https://github.com/vsitzmann/siren). The implementation of inverse rendering task are based on [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch), which is a PyTorch implementation of original [NeRF](https://github.com/bmild/nerf). Thanks to these authors for releasing the code.
