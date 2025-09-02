#!/usr/bin/env python3
"""
Script to collect data from Ball Don't Lie API.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collection import BallDontLieAPI
from config import setup_logging


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Collect NBA data from Ball Don't Lie API"
    )
    parser.add_argument(
        "--seasons", 
        nargs="+",
        type=int, 
        default=[2023, 2024],
        help="Seasons to collect (default: 2023 2024)"
    )
    parser.add_argument(
        "--teams", 
        action="store_true",
        help="Collect team data"
    )
    parser.add_argument(
        "--players", 
        action="store_true",
        help="Collect player data"
    )
    parser.add_argument(
        "--games", 
        action="store_true",
        help="Collect game data"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Collect all data types"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # If --all is specified, collect everything
    if args.all:
        args.teams = args.players = args.games = True
    
    # If nothing specified, collect teams and games
    if not any([args.teams, args.players, args.games]):
        args.teams = args.games = True
    
    logger.info("Starting API data collection")
    logger.info(f"Seasons: {args.seasons}")
    
    try:
        api = BallDontLieAPI()
        
        if args.teams:
            logger.info("Collecting team data...")
            teams_df = api.get_teams()
            logger.info(f"Collected {len(teams_df)} teams")
        
        if args.players:
            logger.info("Collecting player data...")
            players_df = api.get_players()
            logger.info(f"Collected {len(players_df)} players")
        
        if args.games:
            for season in args.seasons:
                logger.info(f"Collecting games for season {season}...")
                games_df = api.get_games(season)
                logger.info(f"Collected {len(games_df)} games for season {season}")
        
        logger.info("API data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error collecting API data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
