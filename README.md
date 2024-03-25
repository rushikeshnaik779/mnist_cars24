# mnist_cars24

Architecture Diagram : 
![Untitled Diagram drawio](https://github.com/rushikeshnaik779/mnist_cars24/assets/34379281/1922e8bc-c4d8-4081-a2d0-98760eca4ee5)


The problem statment: 
Mnist dataset: Predict the digit based on the image provided. 

Folder Structure: 

![Screenshot 2024-03-25 at 11 49 52 AM](https://github.com/rushikeshnaik779/mnist_cars24/assets/34379281/0566e6c6-dfc4-46a4-b166-2a070c92dd89)
- src: folder consist of the files that we will require during the model training. It will contain components like data extraction, feature engineering, model training.
- models: this folder, should not be present here, Since this is a poc, I am trying to save the models here. Ideally it should be in the cloud storage, eg gcs bucket.
- notebooks : this directory will contain all the experiments we have performed in the notebook format, also tutorials to understand usecase for the new onboards.
- backup: As this is in developing state all the unncessary files will go in backup folder, which will be used a as backup incase of a need.
- .gitignore: file helps to ignore unneccesarry file in the remote location, which could take too much of space.
- data : data storage directory. As of now, I am storing here a pkl file of a one instance of a test data, to test the api.
- api : api directory will contain, api require to perform prediction on it. It contains two files, one which invokes all the functions for inference, and another which interact with unvicorn server.
- Dockerfile : it will help to containerize the application as mentioned in the above image.
- main.py: file runs to trigger the training pipeline.
- requirements.txt : file to install all the dependecies
- resources.txt: this file to keep the track of discoveries that I have during the development.  
