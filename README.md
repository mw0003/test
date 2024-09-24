# Neural Network for Baseball Plate Appearance Prediction

This project implements a neural network model to predict the outcomes of baseball plate appearances. The model is based on the approach described in the Baseball Prospectus article [Singlearity: Using a Neural Network to Predict the Outcome of Plate Appearances](https://www.baseballprospectus.com/news/article/59993/singlearity-using-a-neural-network-to-predict-the-outcome-of-plate-appearances/).

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/mw0003/test.git
   cd test
   ```

2. Install the required dependencies:
   ```
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn statsapi
   ```

3. Ensure you have Python 3.7+ installed.

## Usage

1. To train the model and evaluate its performance:
   ```
   python neural_network.py
   ```

2. To input batter and pitcher IDs for predictions:
   ```
   python neural_network.py --predict
   ```
   Follow the prompts to enter batter and pitcher IDs.

3. To run backtesting for the 2024 season:
   ```
   python run_backtest.py
   ```

## Files

- `neural_network.py`: Contains the main neural network model and training logic.
- `data_preprocessing.py`: Handles data loading, preprocessing, and feature engineering.
- `run_backtest.py`: Implements backtesting for the 2024 season.

## Notes

- The model uses data from 2021 to 2024, excluding 2020 due to the shortened season.
- Data is fetched using the MLB-StatsAPI.
- The model predicts 21 possible outcomes for each plate appearance.

## Output

The model will output probabilities for each possible outcome of a plate appearance, given a batter-pitcher matchup.

For any issues or questions, please open an issue in the GitHub repository.
