
# Stock Price Prediction using LSTM

This project implements a Stock Price Prediction model using Long Short-Term Memory (LSTM) neural networks. The model forecasts future stock prices based on historical data, leveraging the Tiingo API to fetch stock prices for analysis. The code normalizes the stock data, trains an LSTM model, and provides predictions for future prices.


## Table of Content 

- Acknowledgement
- API Reference
- Appendix
- Demo
- Deployment
- Environment Variables
- Features
- Related Projects
- License
## Features

- **Historical Data Analysis:** Fetches and analyzes historical stock price data from Tiingo API.
- **LSTM-Based Prediction:** Uses a multi-layered LSTM network for accurate time series forecasting.
- **Future Stock Price Forecasting:** Predicts stock prices for the next 30 days based on the last 100 days of historical data.
- **Visualization:** Generates visual plots comparing predicted vs actual stock prices


## Project Workflow
The project follows a simple, structured workflow for stock price prediction:

- **Data Collection:** Fetch stock price data from Tiingo API using historical data from a specific period.
- **Data Preprocessing:** Normalize the data using MinMaxScaler and split it into training and testing datasets.
- **Model Training:** Use an LSTM-based neural network model to train on historical data (training set).
- **Model Testing:** Test the model on unseen data (testing set) and evaluate its performance using RMSE (Root Mean Squared Error).
- **Prediction:** The trained model is used to predict stock prices for the next 30 days.
- **Visualization:** Plot the actual vs predicted values for both training and testing datasets, along with future predictions.
## Dataset

The stock price data used in this project is fetched using the Tiingo API. The dataset includes:

- Stock Symbol: `AAPL` (Apple Inc.)
- Time Period: From `2020-01-01` to `2023-12-31`
- Data Fields:
  - `Date`: The date of the stock price.
  - `Close`: The closing price of the stock for that day.
  - Other fields can be included like `Open`, `High`, `Low`, and `Volume`.
To use a different stock or time period, you can adjust the API request parameters accordingly.
## Model Architecture
The **LSTM (Long Short-Term Memory)** neural network model used in this project consists of:

- **Input Layer:** Takes input as sequences of historical stock prices.
- **3 LSTM Layers:** Each LSTM layer consists of 50 units, with the first two layers returning sequences to allow for a stacked architecture.
- **Dense Output Layer:** A single neuron output that predicts the next stock price.
- **Loss Function:** Mean Squared Error (MSE).
- **Optimizer:** Adam optimizer to minimize the loss.
The architecture is designed to handle time series data and capture temporal dependencies between stock prices over time.
## Tech Stack
This project leverages the following technologies and tools:

**Python:** Programming language used for scripting.

**TensorFlow & Keras:** For building and training the LSTM model.

**Pandas:** For handling and processing the stock data.

**Numpy:** For numerical computations.

**Matplotlib:** For data visualization and plotting the results.

**Scikit-learn:** For data scaling using MinMaxScaler.

**Tiingo API:** For fetching historical stock price data.


## API Reference

**Tiingo API**

URL: `https://api.tiingo.com/`

Authorization: Requires an API key, passed via headers.

Example Request:
```bash
GET https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate=2020-01-01&endDate=2023-12-31
Authorization: Token YOUR_API_KEY
```## Environment Variables
To run this project, you will need to add the following environment variables to your ```.env``` file:
| Variable            | Description                                                               |
| ----------------- | ------------------------------------------------------------------ |
| ```API_KEY``` | Your Tiingo API key for fetching stock data. |
| ```START_DATE``` | Start date for the stock price data retrieval. |
|```END_DATE```| End date for the stock price data retrieval.|



## Deployment

To deploy the project locally or on a cloud platform:

- Clone the repository:
```bash
  git clone https://github.com/yourusername/stock-price-prediction.git
```

- Install the required dependencies:
```bash
  pip install -r requirements.txt
```
- Ensure that you set your API key as an environment variable (details below).
- Run the main script:
```bash
  python stock_prediction.py
```
## Demo

- **Prediction Plot:** The model visualizes predictions against the actual stock prices.
- **Future Prediction:** Forecasts stock prices for the next 30 days based on the last 100 days of stock data.

To run a demo locally, execute:

```bash
python stock_prediction.py
```

## Results

## Visualization of Data and Model Predictions

### 1. Actual Data vs Train Data vs Test Data
This graph illustrates the comparison between the actual data, the data used for training, and the test data. It helps in understanding how well the model fits the training data and how it generalizes to unseen test data.

![Actual Data, Train Data, Test Data](path/to/screenshot1.png)

### 2. Actual Data vs Predicted Data
In this graph, the actual data is compared with the predicted values from the model. This shows how accurately the model is able to forecast the future based on the input data.

![Actual Data, Predicted Data](path/to/screenshot2.png)

- **Training Data Prediction Plot:** Comparison of predicted vs actual stock prices on the training dataset.
- **Testing Data Prediction Plot:** Comparison of predicted vs actual stock prices on the testing dataset.
- **Future Predictions Plot:** 30-day future stock price forecast based on the model.

## Acknowledgements

 - This project utilizes the  [Tiingo API](https://www.tiingo.com/documentation/general/overview) for fetching stock data.
 - Special thanks to  [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) and [Keras](https://keras.io/api/) communities for providing the LSTM libraries. 

## Appendix

- **LSTM (Long Short-Term Memory)** is a special type of recurrent neural network (RNN) used for time series data.
- **MinMaxScaler** from scikit-learn is used to normalize stock price data between 0 and 1.
- **Data** is split into **training (65%)** and **testing (35%)** sets.

## References

This project is based on the following resources:

1. Tiingo API Documentation: https://api.tiingo.com/
2. LSTM Networks for Time Series Prediction: LSTM theory and applications.
3. TensorFlow/Keras Documentation: For details on how to build LSTM models in TensorFlow.
4. Scikit-learn Documentation: For using MinMaxScaler and other preprocessing techniques.
