import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import statsapi
import logging
from logging.handlers import RotatingFileHandler
import time
import functools
import requests
import os
import json
import statsmodels.api as sm
import shutil
import traceback
from unittest.mock import patch

# Mock for statsapi.get function
def mock_statsapi_get(endpoint, params=None):
    # Add mock implementation here
    return {}  # Return an empty dictionary as a placeholder

# Apply the mock to statsapi.get
statsapi.get = mock_statsapi_get

# Configure logging
log_file = 'data_preprocessing.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Create a file handler that writes to a log file
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Get the root logger and add both handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create a specific logger for API requests
api_logger = logging.getLogger('api_requests')
api_logger.setLevel(logging.DEBUG)
api_file_handler = RotatingFileHandler('api_requests.log', maxBytes=10*1024*1024, backupCount=5)
api_file_handler.setFormatter(log_formatter)
api_logger.addHandler(api_file_handler)

logging.info(f"Logging configured. Debug logs will be written to {os.path.abspath(log_file)}")
logging.info(f"API request logs will be written to {os.path.abspath('api_requests.log')}")

# Add a test log entry
logging.debug("This is a test log entry to verify logging functionality")
api_logger.debug("This is a test log entry for API requests")

def retry_decorator(max_retries=5, initial_delay=1, backoff=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Failed after {max_retries} attempts.")
                        raise
                    logging.warning(f"Attempt {retries} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff
            return None  # This line should never be reached
        return wrapper
    return decorator

def load_mlb_data(years=None, chunk_size=10000):
    import os
    import pickle
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from calendar import monthrange
    from datetime import datetime, date

    logging.info("Entering load_mlb_data function")

    def clean_player_id(player_id):
        try:
            return int(float(player_id))
        except (ValueError, TypeError):
            logging.error(f"Invalid player_id: {player_id}")
            return None

    if years is None:
        years = list(range(2021, 2025))  # Default to years 2021-2024
    elif isinstance(years, int):
        years = [years]

    logging.info(f"Initializing data loading process for years: {years}")

    # Ensure we only process years 2021-2024
    valid_years = set(range(2021, 2025))
    years = sorted(set(years) & valid_years)

    if not years:
        logging.warning("No valid years (2021-2024) specified. Defaulting to all years 2021-2024.")
        years = list(valid_years)

    for year in years:
        logging.info(f"Preparing to process data for year {year}")

    cache_dir = 'mlb_data_cache'
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Cache directory created/verified: {cache_dir}")

    def process_year_data(year):
        logging.info(f"Starting data retrieval for year {year}")
        cache_file = os.path.join(cache_dir, f'mlb_data_{year}.pkl')

        if os.path.exists(cache_file):
            logging.info(f"Loading cached data for year {year}")
            try:
                with open(cache_file, 'rb') as f:
                    year_data = pickle.load(f)
                logging.info(f"Successfully loaded cached data for {year}")
                logging.info(f"Columns in the cached data for {year}: {year_data.columns.tolist()}")

                # Check for the presence of 'earned_runs', 'innings_pitched', and 'runs' columns
                missing_columns = []
                for col in ['earned_runs', 'innings_pitched', 'runs']:
                    if col in year_data.columns:
                        logging.info(f"'{col}' column is present in the cached data for {year}")
                        logging.debug(f"Sample values for '{col}': {year_data[col].head()}")
                    else:
                        logging.warning(f"'{col}' column is missing from the cached data for {year}")
                        missing_columns.append(col)

                if missing_columns:
                    logging.info(f"Retrieving missing columns for {year}: {missing_columns}")
                    year_data = retrieve_missing_columns(year_data, year, missing_columns)

                if 'player_id' not in year_data.columns:
                    logging.error(f"'player_id' column is missing from the cached data for {year}")
                    return None
                if 'opponent_pitcher_id' not in year_data.columns:
                    logging.warning(f"'opponent_pitcher_id' column is missing from the cached data for {year}")
                    year_data = add_opponent_pitcher_id(year_data, year)
                    logging.info(f"Added 'opponent_pitcher_id' column to the cached data for {year}")
                if 'is_home_team' not in year_data.columns:
                    logging.warning(f"'is_home_team' column is missing from the cached data for {year}")
                    year_data['is_home_team'] = year_data['team'] == year_data['home_team']
                    logging.info(f"Added 'is_home_team' column to the cached data for {year}")

                year_data['player_id'] = year_data['player_id'].apply(clean_player_id)
                year_data['opponent_pitcher_id'] = year_data['opponent_pitcher_id'].apply(clean_player_id)
                logging.info(f"Player ID types after cleaning: {year_data['player_id'].dtype}")
                logging.info(f"Opponent pitcher ID types after cleaning: {year_data['opponent_pitcher_id'].dtype}")
                logging.info(f"Number of rows in {year} data: {len(year_data)}")
                logging.debug(f"Sample of cleaned data for {year}:\n{year_data.head()}")
                return year_data
            except Exception as e:
                logging.error(f"Error loading cached data for {year}: {str(e)}")
                logging.debug(f"Traceback: {traceback.format_exc()}")
                if year == 2024:
                    logging.warning(f"Removing corrupted cache file for 2024")
                    os.remove(cache_file)
                return None

        logging.info(f"No cached data found. Starting fresh data retrieval for year {year}")

        # Global dictionary to store information about skipped games
        skipped_games_info = {}

        @retry_decorator(max_retries=3, initial_delay=1, backoff=2)
        def fetch_month_data(month):
            month_data = []
            skipped_games = []
            logging.info(f"Fetching data for {year}-{month:02d}")

            start_date = datetime(year, month, 1)
            _, last_day = monthrange(year, month)
            end_date = datetime(year, month, last_day)

            start_time = time.time()
            max_runtime = 3600  # 1 hour maximum runtime

            try:
                # Fetch all games for the given month
                logging.info(f"Fetching schedule for {year}-{month:02d}")
                games = statsapi.schedule(start_date=start_date.strftime("%Y-%m-%d"),
                                          end_date=end_date.strftime("%Y-%m-%d"))

                if not games:
                    logging.warning(f"No games scheduled for {year}-{month:02d}")
                    return month_data, skipped_games, False

                logging.info(f"Found {len(games)} games for {year}-{month:02d}")

                for game_index, game in enumerate(games):
                    if time.time() - start_time > max_runtime:
                        logging.error(f"Maximum runtime exceeded for {year}-{month:02d}. Processed {game_index} out of {len(games)} games.")
                        break

                    game_id = game['game_id']
                    logging.info(f"Processing game {game_id} ({game_index + 1}/{len(games)})")

                    try:
                        # Fetch detailed game data
                        logging.info(f"Fetching detailed data for game {game_id}")
                        game_data = statsapi.get('game', {'gamePk': game_id})

                        # Initialize players_data list
                        players_data = []

                        # Extract relevant information from game_data
                        home_pitcher_id = None
                        away_pitcher_id = None
                        players_data = []
                        for team in ['home', 'away']:
                            opponent_team = 'away' if team == 'home' else 'home'
                            for player in game_data['gameData']['players'].values():
                                try:
                                    player_data = {
                                        'player_id': clean_player_id(player.get('id')),
                                        'name': player.get('fullName', ''),
                                        'position': player.get('primaryPosition', {}).get('abbreviation', ''),
                                        'team': game_data['gameData']['teams'][team]['name'],
                                        'year': year,
                                        'month': month,
                                        'bats': player.get('batSide', {}).get('code', ''),
                                        'throws': player.get('pitchHand', {}).get('code', ''),
                                        'game_id': game_id,
                                        'age': player.get('currentAge', np.nan),
                                        'height': player.get('height', ''),
                                        'weight': player.get('weight', np.nan),
                                        'jersey_number': player.get('primaryNumber', ''),
                                        'mlb_debut': player.get('mlbDebutDate', ''),
                                        'opponent_team': game_data['gameData']['teams'][opponent_team]['name'],
                                        'venue': game_data['gameData']['venue']['name'],
                                        'game_type': game['game_type'],
                                        'game_date': game['game_date'],
                                        'game_time': game.get('game_time', ''),
                                        'home_team': game_data['gameData']['teams']['home']['name'],
                                        'away_team': game_data['gameData']['teams']['away']['name'],
                                        'is_home_team': team == 'home',
                                        'season': year,
                                        'league': game_data['gameData']['teams'][team].get('league', {}).get('name', ''),
                                        'division': game_data['gameData']['teams'][team].get('division', {}).get('name', ''),
                                        'game_weather': game_data['gameData'].get('weather', {}),
                                        'game_wind': game_data['gameData'].get('wind', {}),
                                        'game_attendance': game_data['gameData'].get('attendance', np.nan),
                                        'game_duration': game_data['gameData'].get('gameDuration', np.nan),
                                        'game_daynight': game_data['gameData'].get('dayNight', ''),
                                        'earned_runs': game_data['liveData']['boxscore']['teams'][team]['teamStats']['pitching'].get('earnedRuns', np.nan),
                                        'innings_pitched': game_data['liveData']['boxscore']['teams'][team]['teamStats']['pitching'].get('inningsPitched', np.nan),
                                        'runs': game_data['liveData']['boxscore']['teams'][team]['teamStats']['batting'].get('runs', np.nan),
                                    }

                                    if player_data['position'] == 'P':
                                        if team == 'home':
                                            home_pitcher_id = player_data['player_id']
                                        else:
                                            away_pitcher_id = player_data['player_id']
                                        player_data['opponent_pitcher_id'] = None
                                    else:
                                        player_data['opponent_pitcher_id'] = away_pitcher_id if team == 'home' else home_pitcher_id

                                    players_data.append(player_data)
                                    logging.debug(f"Processed player {player_data['name']} (ID: {player_data['player_id']})")

                                except Exception as e:
                                    logging.error(f"Error processing player data for game {game_id}: {str(e)}")
                                    continue  # Skip this player and continue with the next one

                        # Verify and correct opponent_pitcher_id assignments
                        for player_data in players_data:
                            if player_data['position'] != 'P' and player_data['opponent_pitcher_id'] is None:
                                player_data['opponent_pitcher_id'] = home_pitcher_id if player_data['team'] == game_data['gameData']['teams']['away']['name'] else away_pitcher_id

                        month_data.extend(players_data)

                        # Add stats (batting, pitching, fielding)
                        for player_data in players_data:
                            if player_data['player_id'] is None:
                                continue  # Skip this player if player_id is None

                            player = game_data['gameData']['players'].get(f"ID{player_data['player_id']}")
                            if player:
                                if 'stats' in player:
                                    for stat_type in ['batting', 'pitching', 'fielding']:
                                        if stat_type in player['stats']:
                                            stats = player['stats'][stat_type]['season']
                                            for stat, value in stats.items():
                                                player_data[f'{stat_type}_{stat}'] = value

                        logging.info(f"Successfully processed game {game_id}")

                        # Implement rate limiting
                        time.sleep(0.3)  # Sleep for 0.3 seconds between API calls

                    except requests.exceptions.Timeout:
                        error_msg = f"Timeout error fetching data for game {game_id}"
                        logging.warning(error_msg)
                        skipped_games.append(game_id)
                    except requests.exceptions.RequestException as req_exc:
                        error_msg = f"Request error fetching data for game {game_id}: {req_exc}"
                        logging.warning(error_msg)
                        skipped_games.append(game_id)
                    except Exception as game_exc:
                        error_msg = f"Unexpected error fetching data for game {game_id}: {game_exc}"
                        logging.error(error_msg)
                        skipped_games.append(game_id)

                    if len(skipped_games) >= 5:  # If 5 consecutive games are skipped, break the loop
                        logging.error(f"Too many consecutive errors for {year}-{month:02d}. Skipping remaining games.")
                        break

            except Exception as exc:
                error_msg = f"Error fetching schedule for {year}-{month:02d}: {exc}"
                logging.error(error_msg)

            if skipped_games:
                logging.warning(f"Skipped {len(skipped_games)} games for {year}-{month:02d}")
                logging.debug(f"Skipped game IDs: {skipped_games}")

            logging.info(f"Fetched {len(month_data)} player records for {year}-{month:02d}")
            return month_data, skipped_games, len(skipped_games) >= 5  # Return a flag indicating if too many errors occurred

        # Generate all months for the year
        date_ranges = range(1, 13)

        # Use ThreadPoolExecutor for parallel processing
        failed_periods = []
        year_data = []
        logging.info(f"Starting parallel processing for {year}")
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(date_ranges))) as executor:
            future_to_date = {executor.submit(fetch_month_data, month): month for month in date_ranges}
            for future in as_completed(future_to_date):
                month = future_to_date[future]
                try:
                    month_data, skipped_games, too_many_errors = future.result()
                    if month_data:
                        year_data.extend(month_data)
                        logging.info(f"Completed data retrieval for {year}-{month:02d}")
                        if skipped_games:
                            logging.warning(f"Skipped {len(skipped_games)} games for {year}-{month:02d}")
                    if too_many_errors:
                        failed_periods.append(month)
                        logging.error(f"Too many errors occurred for {year}-{month:02d}")
                    elif not month_data:
                        failed_periods.append(month)
                        logging.error(f"Failed to retrieve any data for {year}-{month:02d}")
                except Exception as exc:
                    failed_periods.append(month)
                    logging.error(f"Error processing data for {year}-{month:02d}: {exc}")

        if failed_periods:
            logging.warning(f"Failed to retrieve data for {len(failed_periods)} months in {year}: {failed_periods}")

        # Convert the collected player data to a pandas DataFrame
        logging.info(f"Converting collected data to DataFrame for {year}")
        df = pd.DataFrame(year_data)
        logging.info(f"DataFrame created with {len(df)} rows containing player data for {year}.")

        # Check for the presence of 'earned_runs',
        # 'innings_pitched', and 'runs' columns after DataFrame creation
        for col in ['earned_runs', 'innings_pitched', 'runs']:
            if col in df.columns:
                logging.info(f"'{col}' column is present in the initial DataFrame for {year}")
            else:
                logging.warning(f"'{col}' column is missing from the initial DataFrame for {year}")

        # Verify 'player_id' is present in the DataFrame
        if 'player_id' not in df.columns:
            logging.error("'player_id' column is missing from the DataFrame")
            return None
        else:
            logging.info(f"Number of unique player_ids: {df['player_id'].nunique()}")
            logging.info(f"Player ID types: {df['player_id'].dtype}")

        # Verify 'opponent_pitcher_id' is present and correctly populated
        if 'opponent_pitcher_id' not in df.columns:
            logging.error("'opponent_pitcher_id' column is missing from the DataFrame")
            df = add_opponent_pitcher_id(df, year)
        else:
            logging.info(f"Opponent pitcher ID types: {df['opponent_pitcher_id'].dtype}")
            non_pitchers = df[df['position'] != 'P']
            missing_opponent_pitcher = non_pitchers[non_pitchers['opponent_pitcher_id'].isnull()]
            if not missing_opponent_pitcher.empty:
                logging.warning(f"{len(missing_opponent_pitcher)} non-pitcher rows have missing 'opponent_pitcher_id'")
                logging.debug(f"Sample of rows with missing 'opponent_pitcher_id':\n{missing_opponent_pitcher.head()}")
                df = add_opponent_pitcher_id(df, year)
            else:
                logging.info("All non-pitcher rows have 'opponent_pitcher_id' correctly populated")

        # Final check after attempted fixes
        still_missing = df[(df['position'] != 'P') & (df['opponent_pitcher_id'].isnull())]
        if not still_missing.empty:
            logging.error(f"{len(still_missing)} rows still have missing 'opponent_pitcher_id' after fix attempts")
            logging.debug(f"Sample of rows with persistent missing 'opponent_pitcher_id':\n{still_missing.head()}")
            return None
        else:
            logging.info("All 'opponent_pitcher_id' issues have been resolved")

        # Verify uniqueness of opponent_pitcher_id per game
        games_with_multiple_pitchers = df[df['position'] != 'P'].groupby('game_id')['opponent_pitcher_id'].nunique()
        if (games_with_multiple_pitchers > 1).any():
            problematic_games = games_with_multiple_pitchers[games_with_multiple_pitchers > 1].index
            logging.warning(f"Found {len(problematic_games)} games with multiple opponent pitchers assigned")
            logging.debug(f"Problematic game IDs: {problematic_games.tolist()}")
        else:
            logging.info("All games have a unique opponent_pitcher_id assigned to non-pitchers")

        # Handle missing data
        logging.info(f"Handling missing data for {year}")
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column].fillna('Unknown', inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)

        logging.info(f"Missing data handled for {year}.")

        # Log DataFrame info for debugging
        logging.debug(f"DataFrame info:\n{df.info()}")

        # Check for the presence of 'earned_runs', 'innings_pitched', and 'runs' columns after handling missing data
        for col in ['earned_runs', 'innings_pitched', 'runs']:
            if col in df.columns:
                logging.info(f"'{col}' column is present in the final DataFrame for {year}")
                logging.info(f"Number of non-null values in '{col}': {df[col].count()}")
            else:
                logging.warning(f"'{col}' column is missing from the final DataFrame for {year}")

        # Cache the data for the year
        logging.info(f"Caching data for {year}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            logging.info(f"Data successfully cached to {cache_file}")
        except Exception as e:
            logging.error(f"Error caching data for {year}: {str(e)}")

        return df

    def data_generator():
        for year in years:
            year_data = process_year_data(year)
            if year_data is not None:
                for i in range(0, len(year_data), chunk_size):
                    yield year_data.iloc[i:i+chunk_size]

    logging.info("Exiting load_mlb_data function")
    return data_generator()

def retrieve_missing_columns(year_data, year, missing_columns):
    logging.info(f"Retrieving missing columns for {year}: {missing_columns}")

    for game_id in year_data['game_id'].unique():
        game_data = statsapi.get('game', {'gamePk': game_id})

        for team in ['home', 'away']:
            team_stats = game_data['liveData']['boxscore']['teams'][team]['teamStats']

            for col in missing_columns:
                if col == 'earned_runs':
                    value = team_stats['pitching'].get('earnedRuns', np.nan)
                elif col == 'innings_pitched':
                    value = team_stats['pitching'].get('inningsPitched', np.nan)
                elif col == 'runs':
                    value = team_stats['batting'].get('runs', np.nan)
                else:
                    value = np.nan

                year_data.loc[
                    (year_data['game_id'] == game_id) &
                    (year_data['team'] == game_data['gameData']['teams'][team]['name']),
                    col
                ] = value

        logging.info(f"Retrieved missing columns for game {game_id}")

    logging.info(f"Completed retrieval of missing columns for {year}")
    return year_data

def add_opponent_pitcher_id(df, year):
    logging.info(f"Adding 'opponent_pitcher_id' for {year} data")
    for game_id in df['game_id'].unique():
        game_data = df[df['game_id'] == game_id]
        home_pitcher = game_data[(game_data['position'] == 'P') & (game_data['is_home_team'] == True)]['player_id'].values
        away_pitcher = game_data[(game_data['position'] == 'P') & (game_data['is_home_team'] == False)]['player_id'].values

        if len(home_pitcher) > 0 and len(away_pitcher) > 0:
            df.loc[(df['game_id'] == game_id) & (df['is_home_team'] == True) & (df['position'] != 'P'), 'opponent_pitcher_id'] = away_pitcher[0]
            df.loc[(df['game_id'] == game_id) & (df['is_home_team'] == False) & (df['position'] != 'P'), 'opponent_pitcher_id'] = home_pitcher[0]
        else:
            logging.warning(f"Could not find pitchers for game {game_id} in {year}")

    logging.info(f"'opponent_pitcher_id' added for {year} data")
    return df

def calculate_previous_pa_feature(data, feature):
    # Logic to calculate previous plate appearance features
    if feature == 'previous_pa_result':
        return data.groupby('player_id')['result'].shift(1)
    elif feature == 'previous_pa_pitch_count':
        return data.groupby('player_id')['pitch_count'].shift(1)
    elif feature == 'previous_pa_exit_velocity':
        # Use statsmodels.api for exit velocity if available
        try:
            import statsmodels.api as sm
            exit_velocity = data.groupby('player_id')['exit_velocity'].shift(1)
            return sm.nonparametric.lowess(exit_velocity, np.arange(len(exit_velocity)))[:, 1]
        except:
            return data.groupby('player_id')['exit_velocity'].shift(1)
    elif feature == 'previous_pa_launch_angle':
        return data.groupby('player_id')['launch_angle'].shift(1)
    else:
        return np.nan

    # Note: Detailed pitch sequence data has been removed as per instructions

def calculate_pitch_sequence_feature(data, feature):
    # Logic to calculate pitch sequence features
    pitch_number = int(feature.split('_')[1])
    attribute = feature.split('_')[2]

    def get_pitch_attribute(pitch_data, attr):
        if attr == 'type':
            return pitch_data['details']['type']['code'] if 'details' in pitch_data else np.nan
        elif attr == 'velocity':
            return pitch_data['pitchData']['startSpeed'] if 'pitchData' in pitch_data else np.nan
        elif attr == 'movement':
            return (pitch_data['pitchData']['breaks']['breakLength'], pitch_data['pitchData']['breaks']['spinRate']) if 'pitchData' in pitch_data and 'breaks' in pitch_data['pitchData'] else np.nan
        elif attr == 'location':
            return (pitch_data['pitchData']['coordinates']['x'], pitch_data['pitchData']['coordinates']['y']) if 'pitchData' in pitch_data and 'coordinates' in pitch_data['pitchData'] else np.nan
        else:
            return np.nan

    return data['play_by_play'].apply(lambda x: get_pitch_attribute(x['playEvents'][pitch_number-1], attribute) if x and len(x['playEvents']) >= pitch_number else np.nan)

def engineer_features(data):
    logging.info("Starting feature engineering process")
    featured_data = pd.DataFrame()

    # Ensure player_id is included in featured_data
    if 'player_id' in data.columns:
        featured_data['player_id'] = data['player_id']
        logging.info("'player_id' column successfully added to featured_data")
    else:
        logging.error("'player_id' column is missing from the input data")
        raise ValueError("'player_id' column is required for feature engineering")

    # Check for venue_id and add a placeholder if missing
    if 'venue_id' not in data.columns:
        logging.warning("'venue_id' column is missing from the input data. Adding a placeholder.")
        data['venue_id'] = -1  # Use -1 as a placeholder for missing venue_id

    # Define feature groups
    batter_features = [
        'batter_avg', 'batter_obp', 'batter_slg', 'batter_iso', 'batter_bb_rate',
        'batter_k_rate', 'batter_babip', 'batter_woba'
    ]
    pitcher_features = [
        'pitcher_era', 'pitcher_whip', 'pitcher_k_9', 'pitcher_bb_9', 'pitcher_hr_9',
        'pitcher_babip', 'pitcher_lob_pct', 'pitcher_gb_pct'
    ]
    game_situation_features = [
        'inning', 'outs', 'score_difference', 'base_state', 'home_team'
    ]
    team_features = [
        'team_runs_per_game', 'opponent_runs_per_game', 'team_era', 'opponent_era'
    ]
    count_features = ['balls', 'strikes']

    all_features = (batter_features + pitcher_features +
                    game_situation_features + team_features + count_features)

    logging.info(f"Total number of features to engineer: {len(all_features)}")

    # Implement the actual feature engineering logic
    for feature in all_features:
        logging.debug(f"Processing feature: {feature}")
        if feature in data.columns:
            featured_data[feature] = data[feature]
            logging.debug(f"Feature '{feature}' directly copied from input data")
        else:
            logging.debug(f"Calculating derived feature: {feature}")
            try:
                if feature == 'batter_avg':
                    featured_data[feature] = data['batting_hits'] / data['batting_atBats'] if 'batting_hits' in data.columns and 'batting_atBats' in data.columns else np.nan
                elif feature == 'batter_obp':
                    if all(col in data.columns for col in ['batting_hits', 'batting_baseOnBalls', 'batting_hitByPitch', 'batting_atBats', 'batting_sacFlies']):
                        featured_data[feature] = (data['batting_hits'] + data['batting_baseOnBalls'] + data['batting_hitByPitch']) / (data['batting_atBats'] + data['batting_baseOnBalls'] + data['batting_hitByPitch'] + data['batting_sacFlies'])
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'batter_slg':
                    if all(col in data.columns for col in ['batting_hits', 'batting_doubles', 'batting_triples', 'batting_homeRuns', 'batting_atBats']):
                        featured_data[feature] = (data['batting_hits'] + data['batting_doubles'] + 2*data['batting_triples'] + 3*data['batting_homeRuns']) / data['batting_atBats']
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'batter_iso':
                    featured_data[feature] = featured_data['batter_slg'] - featured_data['batter_avg']
                elif feature == 'batter_bb_rate':
                    featured_data[feature] = data['batting_baseOnBalls'] / data['batting_plateAppearances'] if 'batting_baseOnBalls' in data.columns and 'batting_plateAppearances' in data.columns else np.nan
                elif feature == 'batter_k_rate':
                    featured_data[feature] = data['batting_strikeOuts'] / data['batting_plateAppearances'] if 'batting_strikeOuts' in data.columns and 'batting_plateAppearances' in data.columns else np.nan
                elif feature == 'batter_babip':
                    if all(col in data.columns for col in ['batting_hits', 'batting_homeRuns', 'batting_atBats', 'batting_strikeOuts', 'batting_sacFlies']):
                        featured_data[feature] = (data['batting_hits'] - data['batting_homeRuns']) / (data['batting_atBats'] - data['batting_strikeOuts'] - data['batting_homeRuns'] + data['batting_sacFlies'])
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'batter_woba':
                    if all(col in data.columns for col in ['batting_baseOnBalls', 'batting_hitByPitch', 'batting_hits', 'batting_doubles', 'batting_triples', 'batting_homeRuns', 'batting_atBats', 'batting_sacFlies']):
                        featured_data[feature] = (0.69 * data['batting_baseOnBalls'] + 0.72 * data['batting_hitByPitch'] + 0.89 * (data['batting_hits'] - data['batting_doubles'] - data['batting_triples'] - data['batting_homeRuns']) +
                                                  1.27 * data['batting_doubles'] + 1.62 * data['batting_triples'] + 2.10 * data['batting_homeRuns']) / (
                                                  data['batting_atBats'] + data['batting_baseOnBalls'] + data['batting_sacFlies'] + data['batting_hitByPitch'])
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'pitcher_era':
                    featured_data[feature] = (9 * data['pitching_earnedRuns']) / data['pitching_inningsPitched'] if 'pitching_earnedRuns' in data.columns and 'pitching_inningsPitched' in data.columns else np.nan
                elif feature == 'pitcher_whip':
                    featured_data[feature] = (data['pitching_baseOnBalls'] + data['pitching_hits']) / data['pitching_inningsPitched'] if all(col in data.columns for col in ['pitching_baseOnBalls', 'pitching_hits', 'pitching_inningsPitched']) else np.nan
                elif feature == 'pitcher_k_9':
                    featured_data[feature] = (9 * data['pitching_strikeOuts']) / data['pitching_inningsPitched'] if 'pitching_strikeOuts' in data.columns and 'pitching_inningsPitched' in data.columns else np.nan
                elif feature == 'pitcher_bb_9':
                    featured_data[feature] = (9 * data['pitching_baseOnBalls']) / data['pitching_inningsPitched'] if 'pitching_baseOnBalls' in data.columns and 'pitching_inningsPitched' in data.columns else np.nan
                elif feature == 'pitcher_hr_9':
                    featured_data[feature] = (9 * data['pitching_homeRuns']) / data['pitching_inningsPitched'] if 'pitching_homeRuns' in data.columns and 'pitching_inningsPitched' in data.columns else np.nan
                elif feature == 'pitcher_babip':
                    if all(col in data.columns for col in ['pitching_hits', 'pitching_homeRuns', 'pitching_battersFaced', 'pitching_strikeOuts', 'pitching_sacFlies']):
                        featured_data[feature] = (data['pitching_hits'] - data['pitching_homeRuns']) / (data['pitching_battersFaced'] - data['pitching_strikeOuts'] - data['pitching_homeRuns'] + data['pitching_sacFlies'])
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'pitcher_lob_pct':
                    if all(col in data.columns for col in ['pitching_hits', 'pitching_baseOnBalls', 'pitching_hitByPitch', 'pitching_runs', 'pitching_homeRuns']):
                        featured_data[feature] = (data['pitching_hits'] + data['pitching_baseOnBalls'] + data['pitching_hitByPitch'] - data['pitching_runs']) / (
                                                 data['pitching_hits'] + data['pitching_baseOnBalls'] + data['pitching_hitByPitch'] - (1.4 * data['pitching_homeRuns']))
                    else:
                        featured_data[feature] = np.nan
                elif feature == 'pitcher_gb_pct':
                    featured_data[feature] = data['pitching_groundOuts'] / (data['pitching_groundOuts'] + data['pitching_airOuts']) if 'pitching_groundOuts' in data.columns and 'pitching_airOuts' in data.columns else np.nan
                else:
                    featured_data[feature] = np.nan
                    logging.warning(f"Feature '{feature}' could not be calculated directly, set to NaN")

                logging.debug(f"Feature '{feature}' calculated successfully")
            except Exception as e:
                logging.error(f"Error calculating feature '{feature}': {str(e)}")
                featured_data[feature] = np.nan

    # Matchup features calculation removed

    logging.info("Calculating game situation features")
    featured_data = calculate_game_situation_features(featured_data, data)

    logging.info("Calculating team features")
    featured_data = calculate_team_features(featured_data, data)

    # Verify that player_id is still present in the featured_data
    if 'player_id' not in featured_data.columns:
        logging.error("'player_id' column was lost during feature engineering")
        raise ValueError("'player_id' column is missing from the engineered features")

    logging.info(f"Feature engineering complete. Number of features: {len(featured_data.columns)}")
    logging.debug(f"Engineered features: {', '.join(featured_data.columns)}")

    return featured_data

# The calculate_matchup_stats function has been removed as part of
# the effort to eliminate batter-pitcher matchup data from the pipeline.
# This space is intentionally left blank to maintain the overall
# structure of the file and to serve as a reminder that matchup-specific
# functionality has been removed.


# Ensure the api_logger is properly configured
if not api_logger.handlers:
    api_file_handler = RotatingFileHandler('api_requests.log', maxBytes=10*1024*1024, backupCount=5)
    api_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    api_logger.addHandler(api_file_handler)
    api_logger.setLevel(logging.DEBUG)

# Add a console handler for API logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
console_handler.setLevel(logging.INFO)
api_logger.addHandler(console_handler)

# Test API logger
api_logger.debug("API logger configured and tested")

def calculate_game_situation_features(featured_data, data):
    # Some game situation features can be derived from the existing data
    featured_data['home_team'] = (data['team'] == data['home_team']).astype(int)

    def get_base_runners(matchup):
        base_runners = []
        for base in ['postOnFirst', 'postOnSecond', 'postOnThird']:
            runners = matchup.get(base, [])
            if isinstance(runners, dict):
                runners = [runners]
            elif not isinstance(runners, list):
                runners = [runners] if runners else []
            base_runners.extend([
                runner.get('code', '') if isinstance(runner, dict) else str(runner)
                for runner in runners
            ])
        return base_runners

    # Fetch play-by-play data for each game
    for game_id in data['game_id'].unique():
        game_pbp = statsapi.get('game_playByPlay', {'gamePk': game_id})

        # Process play-by-play data to extract game situation features
        for play in game_pbp['allPlays']:
            inning = play['about']['inning']
            outs = play['count']['outs']
            score_difference = play['result'].get('awayScore', 0) - play['result'].get('homeScore', 0)
            base_state = ''.join(sorted(get_base_runners(play['matchup'])))

            # Update featured_data for the corresponding play
            play_index = data[data['game_id'] == game_id].index
            featured_data.loc[play_index, 'inning'] = inning
            featured_data.loc[play_index, 'outs'] = outs
            featured_data.loc[play_index, 'score_difference'] = score_difference
            featured_data.loc[play_index, 'base_state'] = base_state

    # Fetch and add ballpark factors
    ballpark_factors = statsapi.get('venues', {'venueIds': ','.join(data['venue_id'].unique().astype(str))})
    featured_data['ballpark_factor'] = data['venue_id'].map({venue['id']: venue.get('factor', 1.0) for venue in ballpark_factors['venues']})

    # Add weather data (this would require additional API calls or data source)
    # For now, we'll use placeholder values
    featured_data['temperature'] = 70  # placeholder
    featured_data['wind_speed'] = 5    # placeholder
    featured_data['precipitation'] = 0 # placeholder

    return featured_data

def calculate_team_features(featured_data, data):
    # Implement team features calculation
    # This is a placeholder and should be implemented based on available data
    return featured_data

def calculate_last_15_game_stats(featured_data, data):
    for player_id in data['player_id'].unique():
        player_data = data[data['player_id'] == player_id].sort_values('game_date', ascending=False).head(15)

        # Calculate batting stats for last 15 games
        featured_data.loc[data['player_id'] == player_id, 'batter_avg_last_15'] = player_data['hits'].sum() / player_data['at_bats'].sum()
        featured_data.loc[data['player_id'] == player_id, 'batter_obp_last_15'] = (player_data['hits'].sum() + player_data['base_on_balls'].sum() + player_data['hit_by_pitch'].sum()) / (player_data['at_bats'].sum() + player_data['base_on_balls'].sum() + player_data['hit_by_pitch'].sum() + player_data['sacrifice_flies'].sum())
        featured_data.loc[data['player_id'] == player_id, 'batter_slg_last_15'] = (player_data['hits'].sum() + player_data['doubles'].sum() + 2*player_data['triples'].sum() + 3*player_data['home_runs'].sum()) / player_data['at_bats'].sum()
        featured_data.loc[data['player_id'] == player_id, 'batter_woba_last_15'] = (0.69 * player_data['base_on_balls'].sum() + 0.72 * player_data['hit_by_pitch'].sum() + 0.89 * player_data['singles'].sum() +
                                          1.27 * player_data['doubles'].sum() + 1.62 * player_data['triples'].sum() + 2.10 * player_data['home_runs'].sum()) / (
                                          player_data['at_bats'].sum() + player_data['base_on_balls'].sum() + player_data['sacrifice_flies'].sum() + player_data['hit_by_pitch'].sum())

        # Calculate pitching stats for last 15 games
        featured_data.loc[data['player_id'] == player_id, 'pitcher_era_last_15'] = (9 * player_data['earned_runs'].sum()) / player_data['innings_pitched'].sum()
        featured_data.loc[data['player_id'] == player_id, 'pitcher_fip_last_15'] = ((13 * player_data['home_runs'].sum()) + (3 * (player_data['base_on_balls'].sum() + player_data['hit_by_pitch'].sum())) - (2 * player_data['strikeouts'].sum())) / (player_data['innings_pitched'].sum() / 9) + 3.2
        featured_data.loc[data['player_id'] == player_id, 'pitcher_xfip_last_15'] = ((13 * (player_data['fly_balls'].sum() * 0.1)) + (3 * (player_data['base_on_balls'].sum() + player_data['hit_by_pitch'].sum())) - (2 * player_data['strikeouts'].sum())) / (player_data['innings_pitched'].sum() / 9) + 3.2
        featured_data.loc[data['player_id'] == player_id, 'pitcher_k_9_last_15'] = (9 * player_data['strikeouts'].sum()) / player_data['innings_pitched'].sum()

    return featured_data

# The calculate_matchup_stats function has been removed as part of
# the effort to eliminate batter-pitcher matchup data from the pipeline.
# This space is intentionally left blank to maintain the overall
# structure of the file and to serve as a reminder that matchup-specific
# functionality has been removed.

def calculate_game_situation_features(featured_data, data):
    # Some game situation features can be derived from the existing data
    featured_data['home_team'] = (data['team'] == data['home_team']).astype(int)

    # Initialize 'base_state' column as string type
    featured_data['base_state'] = pd.Series(dtype='string')

    def get_base_runners(matchup):
        base_runners = []
        for base in ['postOnFirst', 'postOnSecond', 'postOnThird']:
            runners = matchup.get(base, [])
            if isinstance(runners, dict):
                runners = [runners]
            elif not isinstance(runners, list):
                runners = [runners] if runners else []
            base_runners.extend([
                runner.get('code', '') if isinstance(runner, dict) else str(runner)
                for runner in runners
            ])
        return base_runners

    # Fetch play-by-play data for each game
    for game_id in data['game_id'].unique():
        game_pbp = statsapi.get('game_playByPlay', {'gamePk': game_id})

        # Process play-by-play data to extract game situation features
        for play in game_pbp['allPlays']:
            inning = play['about']['inning']
            outs = play['count']['outs']
            score_difference = play['result'].get('awayScore', 0) - play['result'].get('homeScore', 0)
            base_state = ''.join(sorted(get_base_runners(play['matchup'])))

            # Update featured_data for the corresponding play
            play_index = data[data['game_id'] == game_id].index
            featured_data.loc[play_index, 'inning'] = inning
            featured_data.loc[play_index, 'outs'] = outs
            featured_data.loc[play_index, 'score_difference'] = score_difference
            featured_data.loc[play_index, 'base_state'] = base_state

    # Fetch and add ballpark factors
    venue_ids = ','.join(data['venue_id'].unique().astype(str))
    try:
        ballpark_factors = statsapi.get('venue', {'venueIds': venue_ids})
        if isinstance(ballpark_factors, dict) and 'venues' in ballpark_factors:
            venue_factors = {venue['id']: venue.get('factor', 1.0) for venue in ballpark_factors['venues']}
            featured_data['ballpark_factor'] = data['venue_id'].map(venue_factors)
            logging.info(f"Ballpark factors added for {len(venue_factors)} unique venues")
        else:
            logging.warning("Unexpected response format from venue API. Using default ballpark factor.")
            featured_data['ballpark_factor'] = 1.0
    except Exception as e:
        logging.error(f"Error fetching ballpark factors: {str(e)}")
        featured_data['ballpark_factor'] = 1.0  # Default value if API call fails

    # Log the number of unique ballpark factors actually added
    unique_factors = featured_data['ballpark_factor'].nunique()
    logging.info(f"Number of unique ballpark factors in the data: {unique_factors}")

    # Add weather data (this would require additional API calls or data source)
    # For now, we'll use placeholder values
    featured_data['temperature'] = 70  # placeholder
    featured_data['wind_speed'] = 5    # placeholder
    featured_data['precipitation'] = 0 # placeholder

    # Calculate team and opponent stats
    team_stats = data.groupby('team').agg({
        col: lambda x: pd.to_numeric(x, errors='coerce').mean()
        for col in ['runs', 'earned_runs', 'innings_pitched']
    })

    # Calculate ERA only if necessary columns are present
    if all(col in team_stats.columns for col in ['earned_runs', 'innings_pitched']):
        team_stats['era'] = (team_stats['earned_runs'] * 9) / team_stats['innings_pitched']
    else:
        logging.warning("Unable to calculate ERA due to missing columns. Using placeholder value.")
        team_stats['era'] = np.nan

    # Log the number of non-null values for each column
    for col in team_stats.columns:
        logging.info(f"Number of non-null values in '{col}': {team_stats[col].count()}")

    # Map team stats to featured_data, using np.nan for missing values
    for stat in ['runs', 'era']:
        if stat in team_stats.columns:
            featured_data[f'team_{stat}_per_game'] = featured_data['team'].map(team_stats[stat])
            featured_data[f'opponent_{stat}_per_game'] = featured_data['opponent'].map(team_stats[stat])
        else:
            logging.warning(f"'{stat}' not available. Using placeholder values.")
            featured_data[f'team_{stat}_per_game'] = np.nan
            featured_data[f'opponent_{stat}_per_game'] = np.nan

    return featured_data

# Remove this line as it's causing an undefined name error
# return featured_data[all_features]

def handle_missing_data(data):
    # Impute missing values using mean strategy
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

def normalize_data(data):
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data_normalized

def preprocess_data():
    # Load the data
    data_generator = load_mlb_data()

    processed_chunks = []
    total_rows = 0
    unique_player_ids = set()
    unique_opponent_pitcher_ids = set()

    for chunk in data_generator:
        # Verify 'player_id' and 'opponent_pitcher_id' are present in the chunk
        for field in ['player_id', 'opponent_pitcher_id']:
            if field not in chunk.columns:
                logging.error(f"'{field}' is missing from the chunk data")
                raise ValueError(f"'{field}' column is required but not found in the chunk data")

        logging.info(f"Processing chunk. Shape: {chunk.shape}")

        # Engineer features
        featured_chunk = engineer_features(chunk)

        # Verify 'player_id' and 'opponent_pitcher_id' are still present after feature engineering
        for field in ['player_id', 'opponent_pitcher_id']:
            if field not in featured_chunk.columns:
                logging.error(f"'{field}' was lost during feature engineering")
                raise ValueError(f"'{field}' column is missing after feature engineering")

        # Handle missing data
        imputed_chunk = handle_missing_data(featured_chunk)

        # Normalize the data
        normalized_chunk = normalize_data(imputed_chunk)

        # Update statistics
        total_rows += len(normalized_chunk)
        unique_player_ids.update(normalized_chunk['player_id'])
        unique_opponent_pitcher_ids.update(normalized_chunk['opponent_pitcher_id'])

        processed_chunks.append(normalized_chunk)

    # Concatenate all processed chunks
    normalized_data = pd.concat(processed_chunks, ignore_index=True)

    # Final check for 'player_id' and 'opponent_pitcher_id'
    for field in ['player_id', 'opponent_pitcher_id']:
        if field not in normalized_data.columns:
            logging.error(f"'{field}' is missing from the final normalized data")
            raise ValueError(f"'{field}' column is missing in the final normalized data")

    logging.info(f"Preprocessing complete. Total rows: {total_rows}")
    logging.info(f"Number of unique player_ids in final data: {len(unique_player_ids)}")
    logging.info(f"Number of unique opponent_pitcher_ids in final data: {len(unique_opponent_pitcher_ids)}")

    return normalized_data

def prepare_data_for_model():
    # Preprocess the data
    processed_data = preprocess_data()

    # Define the 21 possible outcomes
    possible_outcomes = [
        'single', 'double', 'triple', 'home_run', 'walk', 'strikeout',
        'hit_by_pitch', 'sacrifice_bunt', 'sacrifice_fly', 'field_out',
        'force_out', 'ground_out', 'fly_out', 'line_out', 'pop_out',
        'intentional_walk', 'double_play', 'triple_play', 'fielders_choice',
        'fielders_choice_out', 'other'
    ]

    # Generate plate_appearance_result from available data
    processed_data['plate_appearance_result'] = processed_data.apply(determine_pa_result, axis=1)

    # Split the data into features (X) and target (y)
    X = processed_data.drop('plate_appearance_result', axis=1)
    y = processed_data['plate_appearance_result']

    # Convert string labels to integer indices
    y = y.map({outcome: i for i, outcome in enumerate(possible_outcomes)})

    # Handle missing values or errors in the target variable
    y = y.fillna(len(possible_outcomes) - 1)  # Assign 'other' to any unexpected outcomes

    # Implement one-hot encoding for the target variable
    from tensorflow.keras.utils import to_categorical
    y_encoded = to_categorical(y, num_classes=len(possible_outcomes))

    # Ensure X and y_encoded have the same number of samples
    if len(X) != len(y_encoded):
        raise ValueError(f"Mismatch in number of samples: X has {len(X)}, y has {len(y_encoded)}")

    return X, y_encoded

def determine_pa_result(row):
    # Logic to determine plate appearance result based on available data
    if row['stats.hits'] > row['stats.hits'].shift(1):
        if row['stats.doubles'] > row['stats.doubles'].shift(1):
            return 'double'
        elif row['stats.triples'] > row['stats.triples'].shift(1):
            return 'triple'
        elif row['stats.homeRuns'] > row['stats.homeRuns'].shift(1):
            return 'home_run'
        else:
            return 'single'
    elif row['stats.baseOnBalls'] > row['stats.baseOnBalls'].shift(1):
        return 'walk'
    elif row['stats.strikeOuts'] > row['stats.strikeOuts'].shift(1):
        return 'strikeout'
    elif row['stats.hitByPitch'] > row['stats.hitByPitch'].shift(1):
        return 'hit_by_pitch'
    elif row['stats.sacBunts'] > row['stats.sacBunts'].shift(1):
        return 'sacrifice_bunt'
    elif row['stats.sacFlies'] > row['stats.sacFlies'].shift(1):
        return 'sacrifice_fly'
    elif row['stats.groundOuts'] > row['stats.groundOuts'].shift(1):
        return 'ground_out'
    elif row['stats.airOuts'] > row['stats.airOuts'].shift(1):
        return 'fly_out'
    else:
        return 'other'

if __name__ == "__main__":
    X, y = prepare_data_for_model()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

def input_matchups():
    matchups = []
    print("Enter batter and pitcher IDs for each matchup (or 'done' to finish):")
    while True:
        batter_id = input("Enter batter ID: ")
        if batter_id.lower() == 'done':
            break
        pitcher_id = input("Enter pitcher ID: ")
        matchups.append((batter_id, pitcher_id))

    matchup_data = []
    for batter_id, pitcher_id in matchups:
        batter_stats = get_player_stats(batter_id, 'hitting')
        pitcher_stats = get_player_stats(pitcher_id, 'pitching')

        matchup_data.append({
            'batter_id': batter_id,
            'pitcher_id': pitcher_id,
            **batter_stats,
            **pitcher_stats
        })

    return pd.DataFrame(matchup_data)

def get_player_stats(player_id, stat_group):
    current_year = datetime.now().year
    stats = statsapi.player_stat_data(player_id, group=stat_group, type='season', season=current_year)
    return stats['stats'][0]['stats'] if stats['stats'] else {}

# The get_matchup_stats function has been removed as part of
# the effort to eliminate batter-pitcher matchup data from the pipeline.
# This space is intentionally left blank to maintain the overall
# structure of the file and to serve as a reminder that matchup-specific
# functionality has been removed.


def output_projected_outcomes(matchup_data, model):
    # Preprocess the matchup data to match the model's input format
    X = preprocess_matchup_data(matchup_data)

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
            'batter_id': matchup_data.iloc[i]['batter_id'],
            'pitcher_id': matchup_data.iloc[i]['pitcher_id'],
            'projected_outcomes': {
                possible_outcomes[j]: float(pred[j]) for j in range(len(possible_outcomes))
            }
        }
        outcomes.append(outcome)

    return outcomes

def preprocess_matchup_data(matchup_data):
    # Ensure all required columns are present
    required_columns = ['batter_id', 'pitcher_id', 'batter_avg', 'batter_obp', 'batter_slg',
                        'pitcher_era', 'pitcher_whip']
    for col in required_columns:
        if col not in matchup_data.columns:
            raise ValueError(f"Required column '{col}' is missing from matchup data")

    # Handle missing values
    matchup_data = handle_missing_data(matchup_data)

    # Engineer features
    featured_data = engineer_features(matchup_data)

    # Normalize the data
    normalized_data = normalize_data(featured_data)

    # Ensure we have 79 input features as expected by the model
    if normalized_data.shape[1] != 79:
        raise ValueError(f"Expected 79 input features, but got {normalized_data.shape[1]}")

    # Convert to numpy array
    return normalized_data.values
