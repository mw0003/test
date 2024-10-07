import statsapi
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='test_api.log',
                    filemode='w')

def test_statsapi_schedule():
    # Set up date range for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Test for the last 7 days

    try:
        logging.info(f"Fetching schedule from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        schedule = statsapi.schedule(start_date=start_date.strftime("%Y-%m-%d"),
                                     end_date=end_date.strftime("%Y-%m-%d"))

        if schedule:
            logging.info(f"Successfully retrieved {len(schedule)} games")
            for game in schedule[:5]:  # Log details of first 5 games
                logging.debug(f"Game ID: {game['game_id']}, Date: {game['game_date']}, Teams: {game['away_name']} at {game['home_name']}")
        else:
            logging.warning("No games found in the specified date range")

        return schedule
    except Exception as e:
        logging.error(f"Error fetching schedule: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_statsapi_schedule()
    if result:
        print(f"Successfully retrieved {len(result)} games. Check test_api.log for details.")
    else:
        print("Failed to retrieve schedule. Check test_api.log for error details.")
