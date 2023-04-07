# IID-MEF
Code of paper IID-MEF: A Multi-exposure Fusion Network Based on Intrinsic Image Decomposition.

#### Recommended Environment:<br>
 - [ ] python = 2.7
 - [ ] tensorflow-gpu = 1.9.0
 - [ ] numpy = 1.15.4
 - [ ] scipy = 1.2.0
 

#### Prepare data :<br>
- [ ] Put multi-exposed images in the "dataset/train/..." for training
- [ ] Put multi-exposed images in the "dataset/test/demo/..." for testing

#### Training :<br>
- [ ] Run "CUDA_VISIBLE_DEVICES=X python IID_Net_train.py" to implement the intrinsic image  decomposition
- [ ] Run "CUDA_VISIBLE_DEVICES=X python R/S/CFus_Net_train.py" to fuse the components.

#### To produce the HDR images :<br>
Run "CUDA_VISIBLE_DEVICES=X python evaluate_Fus.py" to implement MEF, obtaining HDR images.


#### You can visualize the decomposed components by using our IIDNet:<br>
Run "CUDA_VISIBLE_DEVICES=X python evaluate_IID.py" to perform the decomposition.

If this work is helpful to you, please cite it as：
```
@article{zhang2023iid,
  title={IID-MEF: A multi-exposure fusion network based on intrinsic image decomposition},
  author={Zhang, Hao and Ma, Jiayi},
  journal={Information Fusion},
  volume={95},
  pages={326--340},
  year={2023},
  publisher={Elsevier}
}
```
