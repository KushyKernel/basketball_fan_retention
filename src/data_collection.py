"""
Data collection utilities for basketball APIs and web scraping.
"""

import requests
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlencode
import pandas as pd
from bs4 import BeautifulSoup
import logging
# Note: ratelimit package may need to be installed
# from ratelimit import limits, sleep_and_retry

try:
    from .config import load_config, get_data_paths
except ImportError:
    from config import load_config, get_data_paths

logger = logging.getLogger(__name__)


def rate_limit_decorator(calls: int, period: int):
    """Simple rate limiting decorator."""
    last_called = [0.0]
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = period / calls - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


class BallDontLieAPI:
    """Client for the Ball Don't Lie NBA API with caching and rate limiting."""
    
    def __init__(self):
        self.config = load_config()
        api_config = self.config['apis']['ball_dont_lie']
        self.base_url = api_config['base_url']
        self.rate_limit = api_config['rate_limit']
        self.retry_attempts = 3  # Default value
        self.backoff_factor = 1.0  # Default value
        self.cache_dir = get_data_paths()['raw_api']
        
    @rate_limit_decorator(calls=60, period=60)  # 60 calls per minute
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """Make rate-limited API request."""
        url = urljoin(self.base_url, endpoint)
        if params:
            url += '?' + urlencode(params)
            
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url)
                if response.status_code == 429:  # Rate limited
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                wait_time = self.backoff_factor ** attempt
                time.sleep(wait_time)
        
        # This should never be reached due to the raise above, but satisfy type checker
        raise RuntimeError("All retry attempts exhausted")
                
    def _get_cache_key(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for request."""
        cache_string = endpoint
        if params:
            cache_string += json.dumps(params, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get data from API with caching.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use caching
            
        Returns:
            API response data
        """
        cache_key = self._get_cache_key(endpoint, params)
        
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                logger.info(f"Loading from cache: {endpoint}")
                return cached_data
        
        logger.info(f"Making API request: {endpoint}")
        response = self._make_request(endpoint, params)
        data = response.json()
        
        if use_cache:
            self._save_to_cache(cache_key, data)
            
        return data
    
    def get_teams(self) -> pd.DataFrame:
        """Get all NBA teams."""
        data = self.get_data('teams')
        return pd.DataFrame(data['data'])
    
    def get_players(self, per_page: int = 100) -> pd.DataFrame:
        """Get all NBA players with pagination."""
        all_players = []
        page = 1
        
        while True:
            params = {'page': page, 'per_page': per_page}
            data = self.get_data('players', params)
            
            players = data['data']
            if not players:
                break
                
            all_players.extend(players)
            
            if len(players) < per_page:
                break
                
            page += 1
            
        return pd.DataFrame(all_players)
    
    def get_games(self, season: int, team_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get games for a specific season.
        
        Args:
            season: Season year (e.g., 2023 for 2023-24 season)
            team_ids: Optional list of team IDs to filter
            
        Returns:
            DataFrame of games
        """
        all_games = []
        page = 1
        
        while True:
            params = {'seasons[]': season, 'page': page, 'per_page': 100}
            if team_ids:
                for team_id in team_ids:
                    params[f'team_ids[]'] = team_id
                
            data = self.get_data('games', params)
            
            games = data['data']
            if not games:
                break
                
            all_games.extend(games)
            
            if len(games) < 100:
                break
                
            page += 1
            
        return pd.DataFrame(all_games)


class BasketballReferenceScraper:
    """Scraper for Basketball Reference with rate limiting."""
    
    def __init__(self):
        self.config = load_config()
        bbref_config = self.config['apis']['basketball_reference']
        self.base_url = bbref_config['base_url']
        self.rate_limit = bbref_config['rate_limit']
        self.user_agent = 'Basketball Fan Retention Bot 1.0'  # Default user agent
        self.cache_dir = get_data_paths()['raw_bbref']
        
    @rate_limit_decorator(calls=20, period=60)  # 20 calls per minute
    def _make_request(self, url: str) -> requests.Response:
        """Make rate-limited request to Basketball Reference."""
        headers = {'User-Agent': self.user_agent}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response
    
    def _clean_bbref_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Basketball Reference table data."""
        # Remove repeated header rows
        df = df[df.iloc[:, 0] != df.columns[0]]
        
        # Remove footnote symbols
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[*+]', '', regex=True)
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            # Handle thousands separators
            df[col] = df[col].astype(str).str.replace(',', '')
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_team_game_logs(self, team: str, season: int) -> pd.DataFrame:
        """
        Get game logs for a team and season.
        
        Args:
            team: Team abbreviation (e.g., 'LAL')
            season: Season year (e.g., 2024)
            
        Returns:
            DataFrame of game logs
        """
        url = f"{self.base_url}/teams/{team}/{season}/gamelog/"
        
        cache_file = self.cache_dir / f"{team}_{season}_gamelog.csv"
        if cache_file.exists():
            logger.info(f"Loading from cache: {team} {season} game log")
            return pd.read_csv(cache_file)
        
        logger.info(f"Scraping game log: {team} {season}")
        response = self._make_request(url)
        
        # Parse tables with pandas
        tables = pd.read_html(response.text, attrs={'id': 'tgl_basic'})
        if not tables:
            raise ValueError(f"No game log table found for {team} {season}")
        
        df = tables[0]
        df = self._clean_bbref_table(df)
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        
        return df
    
    def get_attendance_data(self, season: int) -> pd.DataFrame:
        """
        Get attendance data for a season.
        
        Args:
            season: Season year
            
        Returns:
            DataFrame of attendance data
        """
        url = f"{self.base_url}/leagues/NBA_{season}_misc.html"
        
        cache_file = self.cache_dir / f"attendance_{season}.csv"
        if cache_file.exists():
            logger.info(f"Loading attendance data from cache: {season}")
            return pd.read_csv(cache_file)
        
        logger.info(f"Scraping attendance data: {season}")
        response = self._make_request(url)
        
        # Parse attendance table
        tables = pd.read_html(response.text, attrs={'id': 'misc_stats'})
        if not tables:
            raise ValueError(f"No attendance table found for {season}")
        
        df = tables[0]
        df = self._clean_bbref_table(df)
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        
        return df
