# Car Prices Prediction

### Dateset:
 We have used the [Vehicle dataset](https://www.kaggle.com/code/mukundiyerw21/vehicle-price-prediction) from Kaggle in this Repo.

### To install the required dependencies run this 
`pip install -r requirements.txt`

### To build the docker image
`docker build -t car_prices_prediction_rf .`

### To push it to docker hub

`docker tag car_prices_prediction_rf nagwa396/car_prices_prediction:latest`

`docker login`

`docker push nagwa396/car_prices_prediction_rf:latest`

### To pull and run image from dockerhub 
`docker pull nagwa396/car_prices_prediction:latest`

`docker run nagwa396/car_prices_prediction:latest`



