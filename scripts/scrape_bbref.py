#!/usr/bin/env python3
"""
Script to scrape data from Basketball Reference.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collection import BasketballReferenceScraper
from config import setup_logging


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Scrape NBA data from Basketball Reference"
    )
    parser.add_argument(
        "--seasons", 
        nargs="+",
        type=int, 
        default=[2023, 2024],
        help="Seasons to scrape (default: 2023 2024)"
    )
    parser.add_argument(
        "--teams", 
        nargs="+",
        default=[
            'ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ],
        help="Team abbreviations to scrape"
    )
    parser.add_argument(
        "--attendance", 
        action="store_true",
        help="Scrape attendance data"
    )
    parser.add_argument(
        "--game-logs", 
        action="store_true",
        help="Scrape team game logs"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Scrape all data types"
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
    
    # If --all is specified, scrape everything
    if args.all:
        args.attendance = args.game_logs = True
    
    # If nothing specified, scrape attendance
    if not any([args.attendance, args.game_logs]):
        args.attendance = True
    
    logger.info("Starting Basketball Reference scraping")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Teams: {len(args.teams)} teams")
    
    try:
        scraper = BasketballReferenceScraper()
        
        if args.attendance:
            for season in args.seasons:
                logger.info(f"Scraping attendance data for season {season}...")
                attendance_df = scraper.get_attendance_data(season)
                logger.info(f"Collected attendance data: {len(attendance_df)} records")
        
        if args.game_logs:
            for season in args.seasons:
                for team in args.teams:
                    logger.info(f"Scraping game logs for {team} {season}...")
                    try:
                        game_log_df = scraper.get_team_game_logs(team, season)
                        logger.info(f"Collected {len(game_log_df)} games for {team}")
                    except Exception as e:
                        logger.warning(f"Failed to scrape {team} {season}: {e}")
        
        logger.info("Basketball Reference scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error scraping Basketball Reference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
