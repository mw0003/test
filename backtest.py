import logging
from logging.handlers import RotatingFileHandler
import tensorflow as tf
from neural_network import create_singlearity_pa_model, custom_baseball_metric
from data_preprocessing import load_mlb_data, prepare_data_for_model
import numpy as np
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = RotatingFileHandler('backtest.log', maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def load_model():
    try:
        model = tf.keras.models.load_model('best_model.h5', custom_objects={'custom_baseball_metric': custom_baseball_metric})
        logger.info("Loaded trained model successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_2024_data(end_date):
    try:
        data_generator = load_mlb_data(years=[2024])
        X, y = prepare_data_for_model(next(data_generator))
        logger.info(f"Loaded 2024 season data up to {end_date}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading 2024 data: {str(e)}")
        raise

def calculate_hit_accuracy(y_true, y_pred):
    hit_indices = [0, 1, 2, 3]  # Indices for single, double, triple, home_run
    y_true_hits = np.sum(y_true[:, hit_indices], axis=1)
    y_pred_hits = np.sum(y_pred[:, hit_indices], axis=1)
    return np.mean(y_true_hits == (y_pred_hits > 0.5))

def backtest_2024_season(model, X_2024, y_2024):
    logger.info("Starting backtesting for 2024 season")
    try:
        # Make predictions
        y_pred = model.predict(X_2024)
        logger.info("Made predictions for 2024 season data")

        # Calculate metrics
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_2024, axis=1))
        hit_accuracy = calculate_hit_accuracy(y_2024, y_pred)

        results = {
            'accuracy': float(accuracy),
            'hit_accuracy': float(hit_accuracy),
        }

        logger.info("Backtesting for 2024 season completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error during backtesting for 2024 season: {str(e)}")
        raise

def run_backtest():
    logger.info("Starting backtesting process")
    try:
        model = load_model()
        end_date = datetime.now()
        X_2024, y_2024 = load_2024_data(end_date)
        results = backtest_2024_season(model, X_2024, y_2024)

        logger.info("Backtesting completed. Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")

        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Detailed results saved in 'backtest_results.json'")

    except Exception as e:
        logger.error(f"Error during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    run_backtest()
