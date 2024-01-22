# PMF-SLAM: Pose-guided and Multi-scale Feature Interaction Based Semantic SLAM for Autonomous Wheel Loader
### State Key Lab of Rail Traffic Control & Safety (BJTU)
This repo contains the source code and dataset for our paper:
<br>
[**PMF-SLAM: Pose-guided and Multi-scale Feature Interaction Based Semantic SLAM for Autonomous Wheel Loader**](https://ieeexplore.ieee.org/document/10405860/)
<br>
[paper](https://ieeexplore.ieee.org/document/10405860/)
![SAMe3d](/Figs/Backbone_work.jpg)


# Step1.Environment Installation
Our paltform configuration: ubuntu18.04, Nvidia RTX 3090, cudatoolkit11.3, python3.7(in anaconda environment).
Note: we didn't testing in other configuration.

## Requirements

Ensure that the installation of the GPU driver(support cuda>=11.3, cudnn) is **completed** before the belows.


- Create a virtual environment and activate it.
```
conda create -n SAMe3d python=3.7
conda activate SAMe3d
```
- Install cudatoolkit(e.g. v11.3) and PyTorch in SAMe3d env.
```
conda install cuda -c nvidia/label/cuda-11.3.0 -c nvidia/label/cuda-11.3.1 -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
<div align=center>
<img src="/Figs/cuda_test.png" width="480" height="128" alt="cuda_test" position="center"/>
</div>

<div align=center>
<img src="/Figs/pytorch_for_cuda.png" width="527" height="340" alt="pytorch" position="center"/>
</div>

- numba, torchpack
```
conda install numba
pip install torchpack
```
- open3d
```
pip install open3d
```
- [spconv](https://github.com/traveller59/spconv) (for cuda version)
<div align=center>
<img src="/Figs/spconv_version.png" width="600" height="360" alt="spconv" position="center"/>
</div>

```
pip install spconv-cu113
```
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
```
conda install pytorch-scatter -c pyg
```
- strictyaml
```
 pip install strictyaml
```
- [nuscene-devkit](https://github.com/nutonomy/nuscenes-devkit)
```
 pip install --no-dependencies nuscenes-devkit==1.1.1
```

# Step2.Data Download and Preparation
We have organized these three datasets. To evaluate/train point cloud, you will need to **download** the required datasets.

- Sany (ours) - [Baidu Drive](https://pan.baidu.com/s/10F5ezH4LgT9glGZ_A16BuQ)(Acquiring copyright)
```
./
├── 
├── ...
└── data_path/
    ├──sany
        ├── Mixing_station(MS)/ # Mixing station scene.       
        │   	└── sequences/
	│		├── 00/ # for training          
	│		│   ├── velodyne/	
	│		|   |	├── xxx.bin
	│		|   |	├── xxx.bin
	│		|   |	└── ...
	│		│   └── labels/ 
	│		|       ├── xxx.label
	│		|       ├── xxx.label
	│		|       └── ...
	│		├── 01/ # for validation
	│		└── 02/ # for testing
        └── points(PG)/ # Proving ground scene.
	   	└── sequences/
			├── 00/ # for training          
			|   └── ...
			├── 01/ # for validation
			└── 02/ # for testing
```
- nuScenes - [Baidu Drive](https://pan.baidu.com/s/1TF80roYGuIm6FhDo0DBmgg?pwd=ai67)(access code: ai67)
```
./
├── 
├── ...
└── data_path/
    ├──nuscenes
        ├── lidarseg/   
        ├── maps/
	├── samples/
        │   └── LIDAR_TOP/	
        |    	├── n008-2018-05-21-11-06-59-0400_LIDAR_TOP_1526915243547836.pcd.bin
        |    	└── ...
	├── v1.0-trainval/
	├── nuscenes_infos_train.pkl/
	├── nuscenes_infos_val.pkl/
        └── nuscenes_infos_test.pkl/
```
- SemanticKITTI - [Baidu Drive](https://pan.baidu.com/s/1LL2LItLEQpOt4HLWodTpWQ?pwd=qaos)(access code: qaos)
```
./
├── 
├── ...
└── data_path/
    ├──sequences
        ├── 00/ # 00-07,09-10 for training          
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        │   ├── velodyne/	
        |    	├── 000000.bin
        |    	├── 000001.bin
        |    	└── ...
        └── 21/
	       └── ...
```

# Step3.Train & Validate

Note: In you virtual env(e.g. SAMe3d) establied from above Step1 to run the belows.
## Pretrained Models
-- We provide a pretrained model [LINK](https://pan.baidu.com/s/15_1Z8cfxTB6reRiA05bG8Q?pwd=wf40) (access code: wf40)

- To train on **Sany-Mixing Station dataset**, run
```
 python train.py --config_path config/sany_mixing_parameters.yaml --device 0
```
- To train on **Sany-Proving ground dataset**, run
```
 python train.py --config_path config/sany_points_parameters.yaml --device 0
```
- To train on **nuScenes dataset**, run
```
 python train_nuscene.py --config_path config/nuScenes.yaml --device 0
```
- To train on **SemanticKITTI dataset**, run
```
 python train.py --config_path config/parameters.yaml --device 0
```

## Acknowledgments
We thanks for the opensource codebases, [Cylinder3D](https://github.com/xinge008/Cylinder3D) and [spconv](https://github.com/traveller59/spconv)
