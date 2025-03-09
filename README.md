
# Stock Price Prediction Using LSTM

This project builds an LSTM (Long Short-Term Memory) neural network to predict stock prices using historical data fetched from Yahoo Finance. The model takes a sequence of past stock prices and forecasts future values.

## Project Overview

- **Stock Symbol:** GOOGL (Alphabet Inc.)
- **Date Range:** Jan 1, 2019 to Dec 31, 2024
- **Model:** LSTM
- **Libraries:** `numpy`, `pandas`, `yfinance`, `matplotlib`, `scikit-learn`, `tensorflow`

## How It Works

1. **Fetch and Prepare Data:**
   - Uses `yfinance` to download historical stock data.
   - Extracts the 'Close' price and reshapes it.

2. **Scale Data:**
   - Normalizes data to a range between 0 and 1 using `MinMaxScaler`.

3. **Create Sequences:**
   - Uses a sliding window approach to create sequences of 60 days of stock prices as input and the next day’s price as the target.

4. **Split Data:**
   - Divides data into an 80% training set and 20% test set.

5. **Build and Train LSTM Model:**
   - A sequential LSTM model with two LSTM layers and a dense output layer.
   - Uses `adam` optimizer and `mean_squared_error` loss function.

6. **Make Predictions:**
   - Predicts stock prices on the test set and scales them back to the original price range.

7. **Visualize Results:**
   - Plots actual vs. predicted prices using `matplotlib`.

## Setup and Execution

```bash
# Clone the repository
git clone https://github.com/your-repo/stock-lstm-prediction.git

# Install dependencies
pip install numpy pandas yfinance matplotlib scikit-learn tensorflow

# Run the script
python stock_prediction.py
```

## Output

### Expected Output Details

- **Actual vs. Predicted Prices:**
  - A line graph comparing the real stock prices of GOOGL to the model’s predicted prices over the test period.
  - The blue line represents actual prices, and the red line shows predicted prices.
  - Visual alignment of the two lines indicates the model’s performance.

- **Training and Validation Loss:**
  - A line graph showing the model’s loss (error) during training and validation.
  - Decreasing loss values suggest the model is learning well.

- **Scaled Price Data:**
  - A visual representation of the normalized stock prices used for training the model.
  - Helps to confirm data scaling and patterns.

## Customization

- Change `stock_symbol` to predict other stocks.
- Adjust `sequence_length` for longer/shorter input sequences.
- Modify model architecture for better performance.

## Future Improvements

- Implement hyperparameter tuning.
- Add more advanced features (like technical indicators).
- Try different models (GRU, Transformer, etc.).

