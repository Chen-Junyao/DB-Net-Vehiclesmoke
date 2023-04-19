# DB-Net: Detecting Vehicle Smoke with Deep Block Networks
Vision-based vehicle smoke detection aims to locate the regions of vehicle smoke in video frames, which plays a vital role in intelligent surveillance. 
Existing methods mainly consider vehicle smoke detection as a problem of bounding-box-based detection or pixel-level semantic segmentation in the deep learning era, which struggle to address the trade-off of localization accuracy and speed. In addition, although various studies have been reported, there is no open benchmark available for real vehicle smoke detection. To address these issues, we propose a Polygon-based annotated Vehicle Smoke Segmentation dataset(PoVSSeg) and a block-wise vehicle smoke detection method.

# __PoVSSeg__ <br>
The PoVSSeg dataset encompasses a wide diversity in terms of road conditions (highway or urban), weather (sunny, cloudy, and rainy), vehicle types (bus, truck, and car), and smoke types which including 3,962 vehicle smoke images with polygon annotations. In addition, the PoVSSeg are randomly splited as traing set and test set, which contained 3812 and 150 images respectly. Figure. 1 illustrates some samples from the PoVSSeg. 
<div align="center">
	<img src="/images/sample.png" alt="Editor" width="550">
</div>
<p align="center">Figure. 1 Several samples in the PoVSSeg. The PoVSSeg contained vehicle smoke images from <br>
various traffic scenes and annotated with the polygon</p> <br>

The structure of the PoSSeg Dataset are as shown:<br>
.<br>
|-------image. <br>
|    &emsp;&emsp;&emsp;   |----img1.png     <br>
|     &emsp;&emsp;&emsp;  |    ...          <br>
|      &emsp;&emsp;&emsp; |----img3962.png  <br>
|-------label    
|      &emsp;&emsp;&emsp; |----label1.png   <br>
|      &emsp;&emsp;&emsp; |    ...          <br>
|      &emsp;&emsp;&emsp; |----label3962.png  <br>
|-------Smoke_splits          <br>
|      &emsp;&emsp;&emsp; |----train.txt      <br>
|      &emsp;&emsp;&emsp; |----val.txt        <br>
|________________________________________________________________________<br>
.<br>
The image "label***.png" is the annotation of "img***.png" <br>
You can download the dataset by filling out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfZ6Pw6muzzNTMrCV5uGrYsLxy0l1veolO-oH70uu1cJp-GUg/viewform?usp=sf_link). 
An email with dataset download link will come to you.


# DB-Net
We propose the DB-Net, a dual-branch network with a main branch and an aggregation branch, for vehicle smoke detection. The main branch is designed for feature extraction, and the aggregation branch is for feature enhancement. With the help of the dual-branch architecture, DB-Net can extract and aggregate features parallelly, which is more efficient than encoder-decoder structured models. The architecture of the DB-Net is presented in Figure 2.
<div align="center">
	<img src="/images/framework_DB-Net.png" alt="Editor" width="700">
</div>
<p align="center">Figure. 2 Overall structure of the DB-Net. The main branch is responsible for extracting features, while the aggregation branch is designed to fuse these features. ’⊕’ represent pixel-wise summation.</p> <br>

Furthermore, we propose a coarse-to-fine training strategy to take full use of existing bounding-box annotated data. Extensive experiments demonstrated the corase-to-fine training strategy boosts our DB-Net effectively.

⭐ DEPLOY get pretrained models from different training strategy:
[Coarse](https://drive.google.com/file/d/1ZRVgfy3ZLv-fD2SIXJCAf7Rr64TW1ck0/view?usp=share_link)、
[Fine](https://drive.google.com/file/d/1CJm2MYgqLvzvgiJ2vkLjv8yPUaYwgYBK/view?usp=share_link)、
[Fine-to-coarse](https://drive.google.com/file/d/1hJmW-HERFbNbNdTsfq1uOqxEG9t3aG6Y/view?usp=share_link)、
[Coarse-to-fine](https://drive.google.com/file/d/1M7PR7yU48SeyJWkhLPFKyk23ilbAtENb/view?usp=share_link), and deploy in real environments

```
python test_val.py
```
get mean Intersection-over-Union results the at block-wise in PoVSSeg test set.

# Citation
Do not forget to cite our work appropriately.

```
@article{chen2023db,
  title={DB-Net: Detecting Vehicle Smoke with Deep Block Networks},
  author={Chen, Junyao and Peng, Xiaojiang},
  journal={Applied Sciences},
  volume={13},
  number={8},
  pages={4941},
  year={2023},
  publisher={MDPI}
}
```
