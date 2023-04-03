# DB-Net: Detecting Vehicle Smoke with Deep Block Networks
Vision-based vehicle smoke detection aims to locate the regions of vehicle smoke in video frames, which plays a vital role in intelligent surveillance. 
Existing methods mainly consider vehicle smoke detection as a problem of bounding-box-based detection or pixel-level semantic segmentation in the deep learning era, which struggle to address the trade-off of localization accuracy and speed. In addition, although various studies have been reported, there is no open benchmark available for real vehicle smoke detection. To address these issues, we propose a Polygon-based annotated Vehicle Smoke Segmentation dataset(PoVSSeg) and a block-wise vehicle smoke detection method.

# __PoVSSeg Dataset__ <br>

The PoVSSeg are randomly splited as traing set and test set, which contained 3812 and 150 images respectly.<br>
The structure of the PoSSeg Dataset are as shown:

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
|    <br>
| The image "label***.png" is the annotation of "img***.png" <br>
|________________________________________________________________________<br>
.<br>

<br>
You can download the dataset by filling out this [form](https://www.baidu.com/). 
An email with dataset download link will come to you.

 [百度一下](https://www.baidu.com/)

# DB-Net
