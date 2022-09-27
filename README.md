# Real-time-detection-of-arrythymia-disease-using-CNN

A real-time arrhythmia detection system based on convolutional neural networks is presented in this work. The flow of the project is shown in the figure given below:

 ![flow](https://user-images.githubusercontent.com/65667280/192426004-fca4e1ae-3a72-4fed-9507-5a063e17b82c.png)

The Andriod app was designed for transferring the ECG signals acquired using the Arduino board to the server. On the server, CNN trained on the MIT-arrhythmia database was used to classify these ECG signals to twelve different types of beats.

Raw ECG Signal

![raw_ECG](https://user-images.githubusercontent.com/65667280/192426783-626d6772-2214-49a1-ad0d-1166cc637132.png)

Filtered ECG signal

![filtered_ECG](https://user-images.githubusercontent.com/65667280/192426824-5fdb198b-a98b-4c83-8837-e3c732f9f1ba.png)

Training accuracy

![training_accuracy](https://user-images.githubusercontent.com/65667280/192426897-918f1d97-af50-4a17-a52c-c34f7b035da4.png)

Confusion matrix

![confusion_matrix](https://user-images.githubusercontent.com/65667280/192426925-afb9b8b1-57c0-4e7f-ba37-0b08842d848d.png)

**Note: Coding was done in Matlab. All the functions required to implement this project are provided in online_detection_arrhythmia.m file.**
