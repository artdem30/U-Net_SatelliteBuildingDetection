# SimpleBuildingDetection-
Trained U-Net model for semantic segmentation of buildings in satellite imagery. Included link to trained weights and training dataset used.

This project was useful in some research I was doing for wildfire risk relating to insurance decisions.

I adapted the example shown by qubvel:
https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

The dataset I used:
https://project.inria.fr/aerialimagelabeling/

Download link to trained model:
https://drive.google.com/file/d/1s7BO9bkH1Bf_aBbCpGZboRGGC0WAjGk6/view?usp=sharing

My trained model achieved .81 IOU Score on satellite imagery from similar locations as the dataset. An average IOU score of 
.72 was achieved for satellite imagery of different residential layouts, such as ones from California. Just pop my trained model into the root folder if you want to use it.

I will not go into much depth on this project because it was just a quick tool that helped me extract residential house coordinates
for another research project I am doing. Hopefully someone finds use in this. I advise visiting qubvel's example for documentation on how to train for multiclass segmentation and the use of their pytorch segmentation library.
