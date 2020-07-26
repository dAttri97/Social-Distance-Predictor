# Social-Distance-Predictor

### A predictor using YoloV3 and OpenCV to classify pedestrians violating the rule of Social Distancing

In the age of global pandemic it is very important to maintain social distancing among citizens. This model helps identify violaters from a CCTV footage or a single image. 
First, a particular frame is feeded to the YoloV3 model to identify humans in that particular frame. The output of the model is then used to calculate the Euclidean distance between to people. If the distance between two people is less than 6 feet then they are classified at violaters and highlighted with a red rectangle.
\
\
It is possible that a person may be a violater in one frame and not in another.
\
\
Examples from the available dataset-
\
### Input Image
![Input](https://github.com/dAttri97/Social-Distance-Predictor/blob/master/pics/pic1.png)

### Output Image
![Input](https://github.com/dAttri97/Social-Distance-Predictor/blob/master/pics/result1.PNG)

### Input Image
![Input](https://github.com/dAttri97/Social-Distance-Predictor/blob/master/pics/pic4.png)

### Output Image
![Input](https://github.com/dAttri97/Social-Distance-Predictor/blob/master/pics/result4.png)
