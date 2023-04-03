# DB-Net: Detecting Vehicle Smoke with Deep Block Networks
Vision-based vehicle smoke detection aims to locate the regions of vehicle smoke in video frames, which plays a vital role in intelligent surveillance. 
Existing methods mainly consider vehicle smoke detection as a problem of bounding-box-based detection or pixel-level semantic segmentation in the deep learning era, which struggle to address the trade-off of localization accuracy and speed. In addition, although various studies have been reported, there is no open benchmark available for real vehicle smoke detection. To address these issues, we propose a Polygon-based annotated Vehicle Smoke Segmentation dataset(PoVSSeg) and a block-wise vehicle smoke detection method.

# __PoVSSeg__ <br>
The PoVSSeg dataset encompasses a wide diversity in terms of road conditions (highway or urban), weather (sunny, cloudy, and rainy), vehicle types (bus, truck, and car), and smoke types which including 3,962 vehicle smoke images with polygon-based annotation. In addition, the PoVSSeg are randomly splited as traing set and test set, which contained 3812 and 150 images respectly. Figure. 1 illustrates some samples from the PoVSSeg. 
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
<div align="center">
	<img src="/images/framework_DB-Net.png" alt="Editor" width="700">
</div>

<p align="center">Figure. 2 Overall structure of the DB-Net. The main branch is responsible for extracting features, while the aggregation branch is designed to fuse these features. ’⊕’  represent pixel-wise summation.</p> <br>


