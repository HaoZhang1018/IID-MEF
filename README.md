# IID-MEF
Code of paper IID-MEF: A Multi-exposure Fusion Network Based on Intrinsic Image Decomposition.

#### Recommended Environment:<br>
 - [ ] python = 2.7
 - [ ] tensorflow-gpu = 1.9.0
 - [ ] numpy = 1.15.4
 - [ ] scipy = 1.2.0
 

#### Prepare data :<br>
Put multi-exposed images in the "dataset/test/demo/..."

#### To produce the HDR images :<br>
Run "CUDA_VISIBLE_DEVICES=0 python evaluate_Fus.py" to implement MEF, obtaining HDR iamges.


#### You can visualize the decomposed components by using our IIDNet:<br>
Run "CUDA_VISIBLE_DEVICES=0 python evaluate_IID.py" to perform the decomposition.

