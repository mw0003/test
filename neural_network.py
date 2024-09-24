import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import plot_model
from data_preprocessing import prepare_data_for_model, normalize_data, load_mlb_data, engineer_features, handle_missing_data, preprocess_matchup_data
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Define class_names globally
class_names = ['single', 'double', 'triple', 'home_run', 'walk', 'strikeout',
               'hit_by_pitch', 'sacrifice_bunt', 'sacrifice_fly', 'field_out',
               'force_out', 'ground_out', 'fly_out', 'line_out', 'pop_out',
               'intentional_walk', 'double_play', 'triple_play', 'fielders_choice',
               'fielders_choice_out', 'other']

def create_singlearity_pa_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(80, activation='relu'),
        layers.Dense(80, activation='relu'),
        layers.Dense(21, activation='softmax')
    ])

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])

    return model

def custom_baseball_metric(y_true, y_pred):
    positive_outcomes = [0, 1, 2, 3, 4, 5, 6]  # Indices for single, double, triple, home_run, walk, hit_by_pitch, intentional_walk
    y_true_positive = tf.reduce_sum(tf.gather(y_true, positive_outcomes, axis=1), axis=1)
    y_pred_positive = tf.reduce_sum(tf.gather(y_pred, positive_outcomes, axis=1), axis=1)

    true_positives = tf.reduce_sum(y_true_positive * y_pred_positive)
    false_negatives = tf.reduce_sum(y_true_positive * (1 - y_pred_positive))
    false_positives = tf.reduce_sum((1 - y_true_positive) * y_pred_positive)

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_score

def calculate_obp(y_true, y_pred):
    on_base_outcomes = [0, 1, 2, 3, 4, 5]  # Indices for single, double, triple, home_run, walk, hit_by_pitch
    y_true_on_base = tf.reduce_sum(tf.gather(y_true, on_base_outcomes, axis=1), axis=1)
    y_pred_on_base = tf.reduce_sum(tf.gather(y_pred, on_base_outcomes, axis=1), axis=1)
    return tf.reduce_mean(y_pred_on_base)

def calculate_slg(y_true, y_pred):
    singles = y_pred[:, 0]
    doubles = y_pred[:, 1] * 2
    triples = y_pred[:, 2] * 3
    home_runs = y_pred[:, 3] * 4
    total_bases = singles + doubles + triples + home_runs
    at_bats = tf.reduce_sum(y_pred[:, :4], axis=1)
    return tf.reduce_mean(total_bases / (at_bats + tf.keras.backend.epsilon()))

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['custom_baseball_metric'], label='Training Custom Metric')
    plt.plot(history.history['val_custom_baseball_metric'], label='Validation Custom Metric')
    plt.title('Custom Baseball Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load and preprocess data
X, y = prepare_data_for_model()

# Ensure we have 79 input features
assert X.shape[1] == 79, f"Expected 79 input features, but got {X.shape[1]}"

# Normalize the data
X = normalize_data(X)

# Split data into train, validation, and test sets based on calendar days
current_date = datetime.now()
test_start_date = current_date - timedelta(days=30)
val_start_date = test_start_date - timedelta(days=30)

X_train = X[X.index < val_start_date]
y_train = y[y.index < val_start_date]
X_val = X[(X.index >= val_start_date) & (X.index < test_start_date)]
y_val = y[(y.index >= val_start_date) & (y.index < test_start_date)]
X_test = X[X.index >= test_start_date]
y_test = y[y.index >= test_start_date]

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Create the model
singlearity_pa = create_singlearity_pa_model(X_train.shape[1])

# Define callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
# Using 200 epochs as per the article, with early stopping to prevent overfitting
# Batch size of 64 is chosen as a balance between computational efficiency and model performance
history = singlearity_pa.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    metrics=[custom_baseball_metric, calculate_obp, calculate_slg]
)

# Plot training history
plot_training_history(history)

# Evaluate the model
test_loss, test_accuracy, top_3_accuracy, custom_metric, obp, slg = singlearity_pa.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Top 3 accuracy: {top_3_accuracy:.4f}")
print(f"Custom baseball metric: {custom_metric:.4f}")
print(f"On-base percentage (OBP): {obp:.4f}")
print(f"Slugging percentage (SLG): {slg:.4f}")

# Calculate accuracy per outcome type
y_pred = singlearity_pa.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

for i, outcome in enumerate(class_names):
    outcome_accuracy = accuracy_score(y_true_classes == i, y_pred_classes == i)
    print(f"Accuracy for {outcome}: {outcome_accuracy:.4f}")

# Calculate overall metrics
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate predictions
y_pred = singlearity_pa.predict(X_test)

# Generate classification report
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

def calculate_woba(y_true, y_pred):
    woba_weights = [0.69, 0.89, 1.27, 1.62, 2.10]  # Weights for BB/HBP, 1B, 2B, 3B, HR
    y_true_woba = np.sum(y_true[:, [4, 5, 0, 1, 2, 3]] * woba_weights, axis=1)
    y_pred_woba = np.sum(y_pred[:, [4, 5, 0, 1, 2, 3]] * woba_weights, axis=1)
    return np.mean(y_pred_woba)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

def backtest_2024_season(model):
    """
    Backtesting function for the 2024 season.

    Args:
    model: Trained neural network model

    Returns:
    Dictionary containing backtesting results
    """
    logging.info("Starting backtesting for 2024 season")
    try:
        # Load 2024 season data, including the most recent data
        current_date = datetime.now()
        X_2024, y_2024 = load_2024_season_data(end_date=current_date)
        logging.info(f"Loaded 2024 season data up to {current_date}")

        # Make predictions
        y_pred = model.predict(X_2024)
        logging.info("Made predictions for 2024 season data")

        # Calculate metrics
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_2024, axis=1))
        hit_accuracy = calculate_hit_accuracy(y_2024, y_pred)
        obp_accuracy = calculate_obp_accuracy(y_2024, y_pred)
        slg_accuracy = calculate_slg_accuracy(y_2024, y_pred)

        # Calculate overall plate appearance outcome prediction
        y_true_classes = np.argmax(y_2024, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)

        # Generate projected outcomes
        projected_outcomes = generate_projected_outcomes(X_2024, y_pred)

        # Calculate additional metrics
        obp_pred = calculate_obp(y_2024, y_pred)
        slg_pred = calculate_slg(y_2024, y_pred)
        woba_pred = calculate_woba(y_2024, y_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        results = {
            'accuracy': accuracy,
            'hit_accuracy': hit_accuracy,
            'obp_accuracy': obp_accuracy,
            'slg_accuracy': slg_accuracy,
            'overall_accuracy': overall_accuracy,
            'projected_outcomes': projected_outcomes,
            'obp_prediction': obp_pred,
            'slg_prediction': slg_pred,
            'woba_prediction': woba_pred,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        logging.info("Backtesting for 2024 season completed successfully")
        return results

    except Exception as e:
        logging.error(f"Error during backtesting for 2024 season: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def load_2024_season_data(end_date):
    """
    Function to load and preprocess 2024 season data up to the specified end date.
    """
    try:
        # Load raw data for 2024 season
        raw_data = load_mlb_data(years=[2024])

        # Filter data up to the end_date
        raw_data = raw_data[raw_data['game_date'] <= end_date]

        # Preprocess the data
        from data_preprocessing import preprocess_data, prepare_data_for_model
        processed_data = preprocess_data(raw_data)

        # Prepare data for model
        X, y_encoded = prepare_data_for_model()

        return X, y_encoded

    except Exception as e:
        logging.error(f"Error loading 2024 season data: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def calculate_hit_accuracy(y_true, y_pred):
    hit_indices = [0, 1, 2, 3]  # Indices for single, double, triple, home_run
    y_true_hits = np.sum(y_true[:, hit_indices], axis=1)
    y_pred_hits = np.sum(y_pred[:, hit_indices], axis=1)
    return np.mean(y_true_hits == (y_pred_hits > 0.5))

def calculate_obp_accuracy(y_true, y_pred):
    obp_indices = [0, 1, 2, 3, 4, 6]  # Indices for single, double, triple, home_run, walk, hit_by_pitch
    y_true_obp = np.sum(y_true[:, obp_indices], axis=1)
    y_pred_obp = np.sum(y_pred[:, obp_indices], axis=1)
    return np.mean(np.abs(y_true_obp - y_pred_obp))

def calculate_slg_accuracy(y_true, y_pred):
    slg_weights = [1, 2, 3, 4]  # Weights for single, double, triple, home_run
    y_true_slg = np.sum(y_true[:, :4] * slg_weights, axis=1)
    y_pred_slg = np.sum(y_pred[:, :4] * slg_weights, axis=1)
    return np.mean(np.abs(y_true_slg - y_pred_slg))

def generate_projected_outcomes(X, y_pred):
    projected_outcomes = []

    for i in range(len(X)):
        outcome = {
            'batter_id': X.iloc[i]['player_id'],
            'pitcher_id': X.iloc[i]['opponent_pitcher_id'],
            'projected_outcomes': {
                class_names[j]: float(y_pred[i, j]) for j in range(len(class_names))
            }
        }
        projected_outcomes.append(outcome)

    return projected_outcomes

def input_batter_pitcher_matchups():
    """
    Function to input a list of batter vs pitcher matchups.
    Retrieves current season stats for batters and pitchers using MLB-StatsAPI.
    Prepares the data for the model.
    """
    import statsapi
    from datetime import datetime
    import pandas as pd
    import logging

    logging.info("Starting input of batter-pitcher matchups")

    matchups = []
    print("Enter batter and pitcher IDs for each matchup (or 'done' to finish):")
    while True:
        batter_id = input("Enter batter ID: ")
        if batter_id.lower() == 'done':
            break
        pitcher_id = input("Enter pitcher ID: ")
        matchups.append((batter_id, pitcher_id))

    current_year = datetime.now().year
    processed_matchups = []

    for batter_id, pitcher_id in matchups:
        try:
            batter_stats = statsapi.player_stat_data(batter_id, group='hitting', type='season', season=current_year)
            pitcher_stats = statsapi.player_stat_data(pitcher_id, group='pitching', type='season', season=current_year)

            if not batter_stats['stats'] or not pitcher_stats['stats']:
                logging.warning(f"No stats found for batter {batter_id} or pitcher {pitcher_id}")
                continue

            matchup_data = {
                'batter_id': batter_id,
                'pitcher_id': pitcher_id,
                'batter_avg': batter_stats['stats'][0]['stats'].get('avg', 0),
                'batter_obp': batter_stats['stats'][0]['stats'].get('obp', 0),
                'batter_slg': batter_stats['stats'][0]['stats'].get('slg', 0),
                'batter_ops': batter_stats['stats'][0]['stats'].get('ops', 0),
                'batter_hr': batter_stats['stats'][0]['stats'].get('homeRuns', 0),
                'batter_rbi': batter_stats['stats'][0]['stats'].get('rbi', 0),
                'pitcher_era': pitcher_stats['stats'][0]['stats'].get('era', 0),
                'pitcher_whip': pitcher_stats['stats'][0]['stats'].get('whip', 0),
                'pitcher_so9': pitcher_stats['stats'][0]['stats'].get('strikeoutsPer9Inn', 0),
                'pitcher_bb9': pitcher_stats['stats'][0]['stats'].get('walksPer9Inn', 0),
                'pitcher_hr9': pitcher_stats['stats'][0]['stats'].get('homeRunsPer9', 0),
            }
            processed_matchups.append(matchup_data)
            logging.info(f"Successfully processed matchup: Batter {batter_id} vs Pitcher {pitcher_id}")
        except Exception as e:
            logging.error(f"Error processing matchup for batter {batter_id} and pitcher {pitcher_id}: {str(e)}")

    if not processed_matchups:
        logging.warning("No valid matchups were processed")
        return pd.DataFrame()

    matchups_df = pd.DataFrame(processed_matchups)
    logging.info(f"Completed input of batter-pitcher matchups. Total matchups: {len(matchups_df)}")
    return matchups_df

def output_projected_matchup_outcomes(matchups, model):
    """
    Function to output projected outcomes for batter vs pitcher matchups.

    Args:
    matchups: DataFrame containing batter vs pitcher matchup data
    model: Trained neural network model

    Returns:
    List of dictionaries containing projected outcomes for each matchup
    """
    from data_preprocessing import preprocess_matchup_data

    # Preprocess the matchup data
    X = preprocess_matchup_data(matchups)

    # Use the model to predict outcomes
    predictions = model.predict(X)

    # Define possible outcomes
    possible_outcomes = [
        'single', 'double', 'triple', 'home_run', 'walk', 'strikeout',
        'hit_by_pitch', 'sacrifice_bunt', 'sacrifice_fly', 'field_out',
        'force_out', 'ground_out', 'fly_out', 'line_out', 'pop_out',
        'intentional_walk', 'double_play', 'triple_play', 'fielders_choice',
        'fielders_choice_out', 'other'
    ]

    # Convert predictions to readable format
    outcomes = []
    for i, pred in enumerate(predictions):
        outcome = {
            'batter_id': matchups.iloc[i]['batter_id'],
            'pitcher_id': matchups.iloc[i]['pitcher_id'],
            'projected_outcomes': {
                possible_outcomes[j]: float(pred[j]) for j in range(len(possible_outcomes))
            }
        }
        outcomes.append(outcome)

    return outcomes
