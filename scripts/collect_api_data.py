#!/usr/bin/env python3
"""
Basketball Fan Retention - NBA API Data Collection Script

This script collects real NBA data from the Ball Don't Lie API to supplement
synthetic data with actual team and player information. The collected data
can be used to enhance the realism of the analysis and provide actual NBA
context for the synthetic fan behavior models.

Data Sources:
- Ball Don't Lie API (https://www.balldontlie.io/)
- Team roster and statistics
- Player performance data
- Game results and scores

Features:
- Automatic rate limiting and retry logic
- Data caching to avoid redundant API calls
- Error handling and recovery mechanisms
- Configurable data collection scope
- Progress tracking and detailed logging

Usage Examples:
    python collect_api_data.py --all                    # Collect all data types
    python collect_api_data.py --teams --games          # Teams and games only
    python collect_api_data.py --seasons 2022 2023 2024 # Specific seasons
    python collect_api_data.py --players --verbose      # Players with debug logs

Author: Basketball Fan Retention Analysis Team
Created: 2024
Last Modified: 2024
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import time

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from data_collection import BallDontLieAPI
from config import setup_logging, get_data_paths


class NBADataCollector:
    """
    NBA data collection manager with enhanced error handling and progress tracking.
    
    Manages the collection of NBA data from external APIs with features like:
    - Rate limiting and retry mechanisms
    - Progress tracking and status reporting  
    - Data validation and quality checks
    - Caching to avoid redundant API calls
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the NBA data collector.
        
        Args:
            verbose (bool): Enable verbose logging for debugging
        """
        self.logger = setup_logging()
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        self.data_paths = get_data_paths()
        self.api_client = None
        self.collection_stats = {
            'start_time': None,
            'end_time': None,
            'data_collected': {},
            'errors_encountered': [],
            'api_calls_made': 0
        }
    
    def collect_all_data(self, seasons: List[int], 
                        include_teams: bool = False,
                        include_players: bool = False, 
                        include_games: bool = False) -> Dict[str, Any]:
        """
        Collect all requested NBA data types.
        
        Args:
            seasons (List[int]): NBA seasons to collect (e.g., [2023, 2024])
            include_teams (bool): Whether to collect team data
            include_players (bool): Whether to collect player data
            include_games (bool): Whether to collect game data
            
        Returns:
            Dict[str, Any]: Summary of data collection results
        """
        self.collection_stats['start_time'] = datetime.now()
        self.logger.info("Starting NBA data collection from Ball Don't Lie API")
        self.logger.info(f"Target seasons: {seasons}")
        self.logger.info(f"Data types: Teams={include_teams}, Players={include_players}, Games={include_games}")
        
        try:
            # Initialize API client
            self.api_client = BallDontLieAPI()
            self.logger.info("API client initialized successfully")
            
            # Collect team data (season-independent)
            if include_teams:
                self._collect_team_data()
            
            # Collect player data (season-independent but large dataset)
            if include_players:
                self._collect_player_data()
            
            # Collect game data for each season
            if include_games:
                for season_year in seasons:
                    self._collect_games_for_season(season_year)
            
            self.collection_stats['end_time'] = datetime.now()
            self._generate_collection_summary()
            
            return self.collection_stats
            
        except Exception as critical_error:
            error_message = f"Critical error during data collection: {str(critical_error)}"
            self.logger.error(error_message)
            self.collection_stats['errors_encountered'].append(error_message)
            raise
    
    def _collect_team_data(self) -> None:
        """Collect NBA team information and roster data."""
        self.logger.info("Collecting NBA team data...")
        
        try:
            teams_data = self.api_client.get_teams()
            self.collection_stats['api_calls_made'] += 1
            
            if teams_data is not None and len(teams_data) > 0:
                # Save team data
                output_path = self.data_paths['raw_api'] / 'nba_teams.csv'
                teams_data.to_csv(output_path, index=False)
                
                teams_count = len(teams_data)
                self.collection_stats['data_collected']['teams'] = {
                    'records': teams_count,
                    'file_path': str(output_path),
                    'collection_time': datetime.now().isoformat()
                }
                
                self.logger.info(f"Successfully collected {teams_count} NBA teams")
                self.logger.info(f"Team data saved to: {output_path}")
                
                # Validate team data
                self._validate_team_data(teams_data)
                
            else:
                warning_msg = "No team data returned from API"
                self.logger.warning(warning_msg)
                self.collection_stats['errors_encountered'].append(warning_msg)
                
        except Exception as team_error:
            error_msg = f"Error collecting team data: {str(team_error)}"
            self.logger.error(error_msg)
            self.collection_stats['errors_encountered'].append(error_msg)
    
    def _collect_player_data(self) -> None:
        """Collect NBA player information and statistics."""
        self.logger.info("Collecting NBA player data...")
        self.logger.warning("Player data collection may take several minutes due to API rate limits")
        
        try:
            players_data = self.api_client.get_players()
            self.collection_stats['api_calls_made'] += 1
            
            if players_data is not None and len(players_data) > 0:
                # Save player data
                output_path = self.data_paths['raw_api'] / 'nba_players.csv'
                players_data.to_csv(output_path, index=False)
                
                players_count = len(players_data)
                self.collection_stats['data_collected']['players'] = {
                    'records': players_count,
                    'file_path': str(output_path),
                    'collection_time': datetime.now().isoformat()
                }
                
                self.logger.info(f"Successfully collected {players_count} NBA players")
                self.logger.info(f"Player data saved to: {output_path}")
                
                # Validate player data
                self._validate_player_data(players_data)
                
            else:
                warning_msg = "No player data returned from API"
                self.logger.warning(warning_msg)
                self.collection_stats['errors_encountered'].append(warning_msg)
                
        except Exception as player_error:
            error_msg = f"Error collecting player data: {str(player_error)}"
            self.logger.error(error_msg)
            self.collection_stats['errors_encountered'].append(error_msg)
    
    def _collect_games_for_season(self, season_year: int) -> None:
        """
        Collect game data for a specific NBA season.
        
        Args:
            season_year (int): NBA season year (e.g., 2024 for 2023-24 season)
        """
        self.logger.info(f"Collecting game data for {season_year-1}-{season_year} NBA season...")
        
        try:
            games_data = self.api_client.get_games(season=season_year)
            self.collection_stats['api_calls_made'] += 1
            
            if games_data is not None and len(games_data) > 0:
                # Save games data with season identifier
                output_path = self.data_paths['raw_api'] / f'nba_games_{season_year}.csv'
                games_data.to_csv(output_path, index=False)
                
                games_count = len(games_data)
                if 'games' not in self.collection_stats['data_collected']:
                    self.collection_stats['data_collected']['games'] = {}
                
                self.collection_stats['data_collected']['games'][season_year] = {
                    'records': games_count,
                    'file_path': str(output_path),
                    'collection_time': datetime.now().isoformat()
                }
                
                self.logger.info(f"Successfully collected {games_count} games for season {season_year}")
                self.logger.info(f"Games data saved to: {output_path}")
                
                # Validate games data
                self._validate_games_data(games_data, season_year)
                
                # Brief delay between season requests to be API-friendly
                if self.collection_stats['api_calls_made'] > 1:
                    self.logger.debug("⏱️  Applying rate limiting delay...")
                    time.sleep(1)
                
            else:
                warning_msg = f"No game data returned for season {season_year}"
                self.logger.warning(warning_msg)
                self.collection_stats['errors_encountered'].append(warning_msg)
                
        except Exception as games_error:
            error_msg = f"Error collecting games for season {season_year}: {str(games_error)}"
            self.logger.error(error_msg)
            self.collection_stats['errors_encountered'].append(error_msg)
    
    def _validate_team_data(self, teams_data) -> None:
        """Validate collected team data for quality and completeness."""
        expected_teams_count = 30  # NBA has 30 teams
        actual_teams_count = len(teams_data)
        
        if actual_teams_count != expected_teams_count:
            warning_msg = f"Expected {expected_teams_count} teams, got {actual_teams_count}"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        
        # Check for required columns
        required_columns = ['id', 'name', 'city', 'abbreviation']
        missing_columns = [col for col in required_columns if col not in teams_data.columns]
        
        if missing_columns:
            warning_msg = f"Missing expected team columns: {missing_columns}"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        else:
            self.logger.debug("Team data validation passed")
    
    def _validate_player_data(self, players_data) -> None:
        """Validate collected player data for quality and completeness."""
        players_count = len(players_data)
        
        # Expect at least 400 active players (rough NBA roster size)
        if players_count < 400:
            warning_msg = f"Unusually low player count: {players_count} (expected ~400+)"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        
        # Check for required columns
        required_columns = ['id', 'first_name', 'last_name', 'team']
        missing_columns = [col for col in required_columns if col not in players_data.columns]
        
        if missing_columns:
            warning_msg = f"Missing expected player columns: {missing_columns}"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        else:
            self.logger.debug("Player data validation passed")
    
    def _validate_games_data(self, games_data, season_year: int) -> None:
        """Validate collected games data for a specific season."""
        games_count = len(games_data)
        
        # NBA regular season has 1,230 games (30 teams × 82 games ÷ 2)
        # Plus playoffs, expect at least 1,200 games per season
        expected_min_games = 1200
        
        if games_count < expected_min_games:
            warning_msg = f"Low game count for season {season_year}: {games_count} (expected {expected_min_games}+)"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        
        # Check for required columns
        required_columns = ['id', 'date', 'home_team', 'visitor_team']
        missing_columns = [col for col in required_columns if col not in games_data.columns]
        
        if missing_columns:
            warning_msg = f"Missing expected game columns: {missing_columns}"
            self.logger.warning(warning_msg)
            self.collection_stats['errors_encountered'].append(warning_msg)
        else:
            self.logger.debug(f"Games data validation passed for season {season_year}")
    
    def _generate_collection_summary(self) -> None:
        """Generate and log a comprehensive summary of the data collection process."""
        start_time = self.collection_stats['start_time']
        end_time = self.collection_stats['end_time']
        duration = (end_time - start_time).total_seconds()
        
        total_records = 0
        for data_type, info in self.collection_stats['data_collected'].items():
            if data_type == 'games':
                # Games data is nested by season
                for season, season_info in info.items():
                    total_records += season_info['records']
            else:
                total_records += info['records']
        
        self.logger.info("NBA Data Collection Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Duration: {duration:.1f} seconds")
        self.logger.info(f"API Calls Made: {self.collection_stats['api_calls_made']}")
        self.logger.info(f"Total Records: {total_records:,}")
        self.logger.info(f"Errors: {len(self.collection_stats['errors_encountered'])}")
        
        # Detail breakdown by data type
        for data_type, info in self.collection_stats['data_collected'].items():
            if data_type == 'games':
                games_total = sum(season_info['records'] for season_info in info.values())
                self.logger.info(f"  Games: {games_total:,} records across {len(info)} seasons")
            else:
                self.logger.info(f"  {data_type.title()}: {info['records']:,} records")
        
        if self.collection_stats['errors_encountered']:
            self.logger.warning("Errors encountered during collection:")
            for error in self.collection_stats['errors_encountered']:
                self.logger.warning(f"  • {error}")
        else:
            self.logger.info("Data collection completed without errors!")


def main():
    """Main function to execute NBA data collection with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Collect NBA data from Ball Don't Lie API for basketball fan retention analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Collect all data types for default seasons
  %(prog)s --teams --games                # Collect teams and games only
  %(prog)s --seasons 2022 2023 2024       # Collect for specific seasons
  %(prog)s --players --verbose            # Collect players with debug output
  %(prog)s --games --seasons 2024         # Collect games for 2024 season only
        """
    )
    
    # Season selection
    parser.add_argument(
        "--seasons", 
        nargs="+",
        type=int, 
        default=[2023, 2024],
        help="NBA seasons to collect (default: 2023 2024). Format: YYYY for season ending in YYYY"
    )
    
    # Data type selection
    parser.add_argument(
        "--teams", 
        action="store_true",
        help="Collect NBA team information and rosters"
    )
    parser.add_argument(
        "--players", 
        action="store_true",
        help="Collect player profiles and statistics (large dataset, may take time)"
    )
    parser.add_argument(
        "--games", 
        action="store_true",
        help="Collect game results and scores for specified seasons"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Collect all data types (teams, players, games)"
    )
    
    # Logging and output options
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging for debugging and progress tracking"
    )
    
    args = parser.parse_args()
    
    # Validate season inputs
    current_year = datetime.now().year
    for season in args.seasons:
        if season < 2000 or season > current_year + 1:
            print(f"ERROR: Invalid season year: {season}. Must be between 2000 and {current_year + 1}")
            sys.exit(1)
    
    # If --all is specified, collect everything
    if args.all:
        args.teams = args.players = args.games = True
    
    # If nothing specified, collect teams and games (most common use case)
    if not any([args.teams, args.players, args.games]):
        args.teams = args.games = True
        print("INFO: No specific data types specified, defaulting to teams and games")
    
    print("NBA Data Collection - Basketball Fan Retention Project")
    print("=" * 60)
    print(f"Target seasons: {args.seasons}")
    print(f"Data types: Teams={args.teams}, Players={args.players}, Games={args.games}")
    if args.verbose:
        print("Verbose logging enabled")
    
    try:
        # Initialize data collector
        collector = NBADataCollector(verbose=args.verbose)
        
        # Execute data collection
        results = collector.collect_all_data(
            seasons=args.seasons,
            include_teams=args.teams,
            include_players=args.players,
            include_games=args.games
        )
        
        # Print final summary
        errors_count = len(results['errors_encountered'])
        if errors_count == 0:
            print(f"\nSUCCESS: NBA data collection completed successfully!")
            print(f"Data saved to: {collector.data_paths['raw_api']}")
        else:
            print(f"\nWARNING: NBA data collection completed with {errors_count} warnings")
            print(f"Data saved to: {collector.data_paths['raw_api']}")
            print(f"Check logs for error details")
        
    except KeyboardInterrupt:
        print(f"\nINFO: Data collection interrupted by user")
        sys.exit(1)
        
    except Exception as critical_error:
        print(f"\nERROR: Critical error during NBA data collection: {str(critical_error)}")
        print("Try running with --verbose for more detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
