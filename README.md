# CHEF: A Pilot Chinese Dataset for Evidence-Based Fact-Checking

This project provides tools for "[CHEF: A Pilot Chinese Dataset for Evidence-Based Fact-Checking.](https://xuminghu.github.io)" in NAACL 2022 as a long paper.


## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

## Usage

Our models are in the Joint directory, and you can also find baseline models under Pipeline directory. We give the specific usage in the corresponding directory.
 

## Data
### Format
```
./data
└── CHEF
    ├── train.json
    ├── dev.json
    └── test.json
```
### Download

* Google Drive: [download](https://drive.google.com/file/d/1QKe9i-yXDKh87p4ukRFSnzE03-hAzMto/view?usp=sharing)<br>
* Tsinghua Cloud: [download](https://cloud.tsinghua.edu.cn/f/a3a1cddc1264445e8178/)<br>
* Baidu Cloud: [download](https://pan.baidu.com/s/1S3RnTsd-YM1dbDaL1CiOMw?pwd=kuwx)<br>

For the Joint model (Ours), you can download the data and put it in the Data directory for use. For the Pipeline model, data needs to be preprocessed, and we give the preprocessed data in the Data directory.
 
## Acknowledgements
[Interpretable_Predictions](https://github.com/bastings/interpretable_predictions)

[Kernel Graph Attention Network](https://github.com/thunlp/KernelGAT)

[X-Fact](https://github.com/utahnlp/x-fact)


## Contact

If you have any problem about our code, feel free to contact: hxm19@mails.tsinghua.edu.cn

## Reference

If the code is used in your research, hope you can cite our paper as follows:
```
@inproceedings{hu2022chef,
  abbr = {NAACL},
  title = {CHEF: A Pilot Chinese Dataset for Evidence-Based Fact-Checking},
  author = {Hu, Xuming and Guo, Zhijiang and Wu, guanyu and Liu, Aiwei and Wen, Lijie and Yu, Philip S.},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2022},
  code = {https://github.com/THU-BPM/CHEF}
}
```