# Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives
The experimental code of "[Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives](https://arxiv.org/abs/xxxx.xxxxx)" in ECCV 2022.

## Abstract
Recently, Implicit Neural Representations (INRs) parameterized by neural networks have emerged as a powerful and promising tool to represent all kinds of signals due to its continuous, differentiable properties, showing many superiorities to classical discretized representations. Nevertheless, training of neural networks for INRs only utilizes input-output pairs, and the derivatives of target output with respect to the input which can be accessed in some cases are usually ignored. In this paper, we propose a training paradigm for INRs whose target output is image pixels, to encode image derivatives in addition to image values within the neural network.
Specifically, we use finite differences to approximate image derivatives.
Further, the neural network activated by ReLUs is poorly suited for representing complex signal's derivatives under the derivative supervision in practice, so 
the periodic activation function is adopted to get better derivative convergence properties. 
Lastly, we show how the training paradigm can be leveraged to solve typical INRs problems, such as image regression, inverse rendering, and demonstrate this training paradigm can improve the data-efficiency and generalization capabilities of INRs.

<img src='imgs/pipeline.png'/>

## Setup

### Environment

* Clone this repo
    ```shell
    git clone https://github.com/prstrive/Sobolev_training_INRs.git
    cd Sobolev_training_INRs
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
