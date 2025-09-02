"""
HYPER-REALISTIC synthetic data generation for basketball fan retention analysis.

This module creates realistic synthetic data by modeling:

1. Basketball-specific seasonality patterns
2. Team-specific fan loyalty and market dynamics  
3. Economic factors (recession, inflation, unemployment)
4. Superstar player effects and trades
5. Championship dynasties and bandwagon effects
6. Social influence networks and viral moments
7. Marketing campaign effectiveness
8. Streaming vs traditional media consumption
9. Weather and geographic impacts
10. Mobile app behavioral patterns
11. Competitive sports landscape
12. Real-world churn psychology
13. Pricing psychology and elasticity
14. Generational technology adoption curves
15. Injury impact on star players and fan engagement
16. Draft lottery and trade deadline effects
17. International player popularity (Giannis, Luka, etc.)
18. Gambling/fantasy sports integration
19. Social media influencer effects
20. Real-world attendance constraints and capacity limits
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import random

try:
    from .config import load_config, get_data_paths
except ImportError:
    from config import load_config, get_data_paths

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN SYNTHETIC DATA GENERATOR CLASS
# ============================================================================

class UltraRealisticSyntheticDataGenerator:
    """
    Generate extremely realistic synthetic basketball fan data with advanced behavioral modeling.
    
    This class creates synthetic data that closely mimics real-world basketball fan behavior
    by incorporating over 20 realistic factors including economics, team performance,
    seasonality, demographics, and social influences.
    
    Key Features:
    - Ultra-realistic customer segmentation (casual, regular, avid, super_fan)
    - Economic timeline modeling (recessions, inflation, unemployment)
    - Team-specific loyalty and market dynamics
    - Superstar player impact and championship effects
    - Advanced churn psychology modeling
    - Multi-generational technology adoption patterns
    
    Args:
        random_seed (Optional[int]): Seed for reproducible data generation
        
    Example:
        generator = UltraRealisticSyntheticDataGenerator(random_seed=42)
        data = generator.generate_all_data()
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the synthetic data generator with configuration and random seed."""
        self.config = load_config()
        synthetic_config = self.config['data']['synthetic']
        self.num_customers = synthetic_config['n_customers']
        
        # Calculate months from date range
        start_date = pd.to_datetime(synthetic_config['date_range']['start'])
        end_date = pd.to_datetime(synthetic_config['date_range']['end'])
        self.num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        self.start_date = start_date
        self.end_date = end_date
        
        self.random_seed = random_seed or 42
        self.data_paths = get_data_paths()
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # ================================================================
        # ECONOMIC TIMELINE DATA - Real-world economic events and impacts
        # ================================================================
        self.economic_timeline = {
            '2021-01': {'recession_factor': 1.0, 'inflation': 0.02, 'unemployment': 0.063},
            '2021-06': {'recession_factor': 0.95, 'inflation': 0.025, 'unemployment': 0.055},  # COVID recovery
            '2022-01': {'recession_factor': 1.05, 'inflation': 0.07, 'unemployment': 0.04},   # Inflation spike
            '2022-06': {'recession_factor': 1.0, 'inflation': 0.08, 'unemployment': 0.035},
            '2023-01': {'recession_factor': 0.98, 'inflation': 0.06, 'unemployment': 0.037},  # Mild recession
            '2023-06': {'recession_factor': 0.95, 'inflation': 0.04, 'unemployment': 0.042},
            '2024-01': {'recession_factor': 1.02, 'inflation': 0.03, 'unemployment': 0.038},  # Recovery
            '2024-06': {'recession_factor': 1.05, 'inflation': 0.025, 'unemployment': 0.035}, # Economic growth
            '2025-01': {'recession_factor': 1.08, 'inflation': 0.022, 'unemployment': 0.033}, # Strong economy
            '2025-06': {'recession_factor': 1.1, 'inflation': 0.02, 'unemployment': 0.032}   # Current period
        }
        
        # Superstar players and their impact periods
        self.superstar_timeline = {
            'LeBron James': {'teams': ['LAL'], 'peak_years': ['2021', '2022'], 'decline_years': ['2023', '2024', '2025']},
            'Stephen Curry': {'teams': ['GSW'], 'peak_years': ['2021', '2022', '2023'], 'decline_years': ['2024', '2025']},
            'Kevin Durant': {'teams': ['BRK', 'PHX'], 'peak_years': ['2021', '2022', '2023'], 'decline_years': ['2024', '2025']},
            'Giannis Antetokounmpo': {'teams': ['MIL'], 'peak_years': ['2021', '2022', '2023', '2024'], 'decline_years': []},
            'Luka Doncic': {'teams': ['DAL'], 'peak_years': ['2022', '2023', '2024', '2025'], 'decline_years': []},
            'Jayson Tatum': {'teams': ['BOS'], 'peak_years': ['2022', '2023', '2024', '2025'], 'decline_years': []},
            'Nikola Jokic': {'teams': ['DEN'], 'peak_years': ['2021', '2022', '2023', '2024'], 'decline_years': []},
            'Joel Embiid': {'teams': ['PHI'], 'peak_years': ['2021', '2022', '2023', '2024'], 'decline_years': []},
            'Jimmy Butler': {'teams': ['MIA'], 'peak_years': ['2021', '2022'], 'decline_years': ['2023', '2024', '2025']},
            'Kawhi Leonard': {'teams': ['LAC'], 'peak_years': ['2021'], 'decline_years': ['2022', '2023', '2024', '2025']},
            'Victor Wembanyama': {'teams': ['SA'], 'peak_years': ['2024', '2025'], 'decline_years': []},
            'Shai Gilgeous-Alexander': {'teams': ['OKC'], 'peak_years': ['2024', '2025'], 'decline_years': []}
        }
        
        # Championship timeline and dynasty effects
        self.championship_timeline = {
            '2021': {'champion': 'MIL', 'runner_up': 'PHX', 'cinderella': ['ATL'], 'disappointments': ['BRK', 'LAL']},
            '2022': {'champion': 'GSW', 'runner_up': 'BOS', 'cinderella': ['MIA'], 'disappointments': ['PHX', 'MIL']},
            '2023': {'champion': 'DEN', 'runner_up': 'MIA', 'cinderella': ['MIA'], 'disappointments': ['BOS', 'PHI']},
            '2024': {'champion': 'BOS', 'runner_up': 'DAL', 'cinderella': ['IND'], 'disappointments': ['DEN', 'MIN']},
            '2025': {'champion': 'TBD', 'contenders': ['BOS', 'OKC', 'DEN', 'NYK'], 'dark_horses': ['ORL', 'MEM']}
        }
        
        # Social media viral moments and cultural events
        self.viral_moments = {
            '2021-02': ['LeBron All-Star snub controversy', 'Zion weight concerns viral'],
            '2021-07': ['Giannis championship speech', 'Chris Paul Finals heartbreak'],
            '2022-03': ['Will Smith slap affects celebrity attendance', 'March Madness overlap'],
            '2022-06': ['Warriors dynasty comeback narrative', 'Celtics underdog story'],
            '2023-01': ['Tiktok NBA content explosion', 'Streaming vs cable debate'],
            '2023-06': ['Jimmy Butler playoff legend memes', 'Jokic horse racing viral'],
            '2024-02': ['All-Star weekend Las Vegas spectacle', 'Trade deadline chaos'],
            '2024-06': ['Celtics banner 18 celebration', 'Kyrie Irving redemption arc'],
            '2024-10': ['Bronny James debut hype', 'Victor Wembanyama sophomore surge'],
            '2025-02': ['Cooper Flagg draft hype builds', 'AI referee controversy'],
            '2025-06': ['Streaming wars intensify', 'Gen Z fan engagement revolution']
        }
        
        # Streaming platform competition
        self.streaming_landscape = {
            '2021': {'cable_dominance': 0.7, 'streaming_adoption': 0.3, 'illegal_streams': 0.15},
            '2022': {'cable_dominance': 0.65, 'streaming_adoption': 0.35, 'illegal_streams': 0.18},
            '2023': {'cable_dominance': 0.6, 'streaming_adoption': 0.4, 'illegal_streams': 0.22},
            '2024': {'cable_dominance': 0.55, 'streaming_adoption': 0.45, 'illegal_streams': 0.25},
            '2025': {'cable_dominance': 0.5, 'streaming_adoption': 0.5, 'illegal_streams': 0.28}
        }
        
        # Weather patterns affecting attendance (realistic US climate data)
        self.weather_patterns = {
            'northeast': {'winter_impact': 0.85, 'summer_boost': 1.05},
            'southeast': {'winter_impact': 1.02, 'summer_boost': 0.95, 'hurricane_season': 0.9},
            'midwest': {'winter_impact': 0.8, 'summer_boost': 1.1, 'tornado_season': 0.95},
            'southwest': {'winter_impact': 1.1, 'summer_boost': 0.85},
            'west': {'winter_impact': 1.0, 'summer_boost': 0.98, 'wildfire_season': 0.92}
        }

        # ================================================================
        # NBA TEAM CONFIGURATIONS - Detailed team characteristics and market data
        # ================================================================
        # Each team includes: market size, recent success, fan loyalty, ticket prices,
        # social media following, bandwagon factor, celebrity endorsements, etc.
        self.nba_teams = {
            'ATL': {
                'city': 'Atlanta', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.6,
                'media_market_rank': 8, 'ticket_prices': 'medium', 'arena_capacity': 16600,
                'social_media_following': 850000, 'bandwagon_factor': 0.4, 'local_competition': ['Falcons', 'Braves'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['2Chainz', 'Quavo']
            },
            'BOS': {
                'city': 'Boston', 'market_size': 'large', 'recent_success': 'high', 'fanbase_loyalty': 0.95,
                'media_market_rank': 10, 'ticket_prices': 'high', 'arena_capacity': 19156,
                'social_media_following': 2100000, 'bandwagon_factor': 0.2, 'local_competition': ['Patriots', 'Red Sox'],
                'ownership_stability': 0.95, 'front_office_reputation': 0.9, 'celebrity_fans': ['Ben Affleck', 'Mark Wahlberg']
            },
            'BRK': {
                'city': 'Brooklyn', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.4,
                'media_market_rank': 1, 'ticket_prices': 'very_high', 'arena_capacity': 17732,
                'social_media_following': 1200000, 'bandwagon_factor': 0.7, 'local_competition': ['Knicks', 'Yankees'],
                'ownership_stability': 0.6, 'front_office_reputation': 0.5, 'celebrity_fans': ['Jay-Z', 'Spike Lee']
            },
            'CHA': {
                'city': 'Charlotte', 'market_size': 'medium', 'recent_success': 'low', 'fanbase_loyalty': 0.7,
                'media_market_rank': 24, 'ticket_prices': 'low', 'arena_capacity': 19077,
                'social_media_following': 450000, 'bandwagon_factor': 0.3, 'local_competition': ['Panthers'],
                'ownership_stability': 0.9, 'front_office_reputation': 0.6, 'celebrity_fans': ['Michael Jordan']
            },
            'CHI': {
                'city': 'Chicago', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.85,
                'media_market_rank': 3, 'ticket_prices': 'high', 'arena_capacity': 20917,
                'social_media_following': 1800000, 'bandwagon_factor': 0.3, 'local_competition': ['Bears', 'Cubs'],
                'ownership_stability': 0.7, 'front_office_reputation': 0.6, 'celebrity_fans': ['Barack Obama', 'Chance the Rapper']
            },
            'CLE': {
                'city': 'Cleveland', 'market_size': 'medium', 'recent_success': 'medium', 'fanbase_loyalty': 0.9,
                'media_market_rank': 19, 'ticket_prices': 'low', 'arena_capacity': 19432,
                'social_media_following': 900000, 'bandwagon_factor': 0.4, 'local_competition': ['Browns', 'Guardians'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['Machine Gun Kelly']
            },
            'DAL': {
                'city': 'Dallas', 'market_size': 'large', 'recent_success': 'high', 'fanbase_loyalty': 0.7,
                'media_market_rank': 5, 'ticket_prices': 'medium', 'arena_capacity': 19200,
                'social_media_following': 1100000, 'bandwagon_factor': 0.5, 'local_competition': ['Cowboys', 'Rangers'],
                'ownership_stability': 0.95, 'front_office_reputation': 0.8, 'celebrity_fans': ['Mark Cuban', 'Usher']
            },
            'DEN': {
                'city': 'Denver', 'market_size': 'medium', 'recent_success': 'high', 'fanbase_loyalty': 0.8,
                'media_market_rank': 16, 'ticket_prices': 'medium', 'arena_capacity': 19520,
                'social_media_following': 750000, 'bandwagon_factor': 0.4, 'local_competition': ['Broncos', 'Rockies'],
                'ownership_stability': 0.85, 'front_office_reputation': 0.85, 'celebrity_fans': ['Chauncey Billups']
            },
            'DET': {
                'city': 'Detroit', 'market_size': 'medium', 'recent_success': 'low', 'fanbase_loyalty': 0.75,
                'media_market_rank': 14, 'ticket_prices': 'low', 'arena_capacity': 20491,
                'social_media_following': 600000, 'bandwagon_factor': 0.2, 'local_competition': ['Lions', 'Tigers'],
                'ownership_stability': 0.7, 'front_office_reputation': 0.6, 'celebrity_fans': ['Eminem', 'Big Sean']
            },
            'GSW': {
                'city': 'Golden State', 'market_size': 'large', 'recent_success': 'high', 'fanbase_loyalty': 0.65,
                'media_market_rank': 6, 'ticket_prices': 'very_high', 'arena_capacity': 18064,
                'social_media_following': 3200000, 'bandwagon_factor': 0.8, 'local_competition': ['49ers', 'Giants'],
                'ownership_stability': 0.9, 'front_office_reputation': 0.95, 'celebrity_fans': ['E-40', 'MC Hammer']
            },
            'HOU': {
                'city': 'Houston', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.6,
                'media_market_rank': 7, 'ticket_prices': 'medium', 'arena_capacity': 18055,
                'social_media_following': 800000, 'bandwagon_factor': 0.4, 'local_competition': ['Texans', 'Astros'],
                'ownership_stability': 0.75, 'front_office_reputation': 0.7, 'celebrity_fans': ['Travis Scott', 'Slim Thug']
            },
            'IND': {
                'city': 'Indiana', 'market_size': 'medium', 'recent_success': 'medium', 'fanbase_loyalty': 0.85,
                'media_market_rank': 26, 'ticket_prices': 'low', 'arena_capacity': 17923,
                'social_media_following': 550000, 'bandwagon_factor': 0.2, 'local_competition': ['Colts'],
                'ownership_stability': 0.9, 'front_office_reputation': 0.8, 'celebrity_fans': ['Larry Bird']
            },
            'LAC': {
                'city': 'LA Clippers', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.35,
                'media_market_rank': 2, 'ticket_prices': 'very_high', 'arena_capacity': 19068,
                'social_media_following': 900000, 'bandwagon_factor': 0.8, 'local_competition': ['Lakers', 'Dodgers'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['Steve Ballmer']
            },
            'LAL': {
                'city': 'LA Lakers', 'market_size': 'large', 'recent_success': 'high', 'fanbase_loyalty': 0.8,
                'media_market_rank': 2, 'ticket_prices': 'very_high', 'arena_capacity': 19068,
                'social_media_following': 4500000, 'bandwagon_factor': 0.7, 'local_competition': ['Clippers', 'Dodgers'],
                'ownership_stability': 0.85, 'front_office_reputation': 0.75, 'celebrity_fans': ['Jack Nicholson', 'Rihanna']
            },
            'MEM': {
                'city': 'Memphis', 'market_size': 'small', 'recent_success': 'medium', 'fanbase_loyalty': 0.85,
                'media_market_rank': 51, 'ticket_prices': 'low', 'arena_capacity': 17794,
                'social_media_following': 400000, 'bandwagon_factor': 0.3, 'local_competition': ['None'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.8, 'celebrity_fans': ['Justin Timberlake']
            },
            'MIA': {
                'city': 'Miami', 'market_size': 'large', 'recent_success': 'high', 'fanbase_loyalty': 0.5,
                'media_market_rank': 18, 'ticket_prices': 'high', 'arena_capacity': 19600,
                'social_media_following': 1500000, 'bandwagon_factor': 0.8, 'local_competition': ['Dolphins', 'Marlins'],
                'ownership_stability': 0.95, 'front_office_reputation': 0.9, 'celebrity_fans': ['DJ Khaled', 'Rick Ross']
            },
            'MIL': {
                'city': 'Milwaukee', 'market_size': 'medium', 'recent_success': 'high', 'fanbase_loyalty': 0.95,
                'media_market_rank': 35, 'ticket_prices': 'medium', 'arena_capacity': 17500,
                'social_media_following': 700000, 'bandwagon_factor': 0.3, 'local_competition': ['Packers', 'Brewers'],
                'ownership_stability': 0.9, 'front_office_reputation': 0.85, 'celebrity_fans': ['Aaron Rodgers']
            },
            'MIN': {
                'city': 'Minnesota', 'market_size': 'medium', 'recent_success': 'low', 'fanbase_loyalty': 0.75,
                'media_market_rank': 15, 'ticket_prices': 'medium', 'arena_capacity': 19356,
                'social_media_following': 500000, 'bandwagon_factor': 0.3, 'local_competition': ['Vikings', 'Twins'],
                'ownership_stability': 0.7, 'front_office_reputation': 0.6, 'celebrity_fans': ['Prince Estate']
            },
            'NOP': {
                'city': 'New Orleans', 'market_size': 'small', 'recent_success': 'medium', 'fanbase_loyalty': 0.7,
                'media_market_rank': 50, 'ticket_prices': 'low', 'arena_capacity': 16867,
                'social_media_following': 450000, 'bandwagon_factor': 0.4, 'local_competition': ['Saints'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['Lil Wayne']
            },
            'NYK': {
                'city': 'New York', 'market_size': 'large', 'recent_success': 'low', 'fanbase_loyalty': 0.8,
                'media_market_rank': 1, 'ticket_prices': 'very_high', 'arena_capacity': 19812,
                'social_media_following': 2800000, 'bandwagon_factor': 0.4, 'local_competition': ['Nets', 'Yankees'],
                'ownership_stability': 0.6, 'front_office_reputation': 0.4, 'celebrity_fans': ['Spike Lee', 'Tracy Morgan']
            },
            'OKC': {
                'city': 'Oklahoma City', 'market_size': 'small', 'recent_success': 'medium', 'fanbase_loyalty': 0.95,
                'media_market_rank': 45, 'ticket_prices': 'low', 'arena_capacity': 18203,
                'social_media_following': 600000, 'bandwagon_factor': 0.2, 'local_competition': ['Sooners'],
                'ownership_stability': 0.85, 'front_office_reputation': 0.85, 'celebrity_fans': ['Blake Shelton']
            },
            'ORL': {
                'city': 'Orlando', 'market_size': 'medium', 'recent_success': 'low', 'fanbase_loyalty': 0.6,
                'media_market_rank': 20, 'ticket_prices': 'low', 'arena_capacity': 18846,
                'social_media_following': 500000, 'bandwagon_factor': 0.3, 'local_competition': ['Magic', 'UCF'],
                'ownership_stability': 0.75, 'front_office_reputation': 0.6, 'celebrity_fans': ['Dwight Howard']
            },
            'PHI': {
                'city': 'Philadelphia', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.85,
                'media_market_rank': 4, 'ticket_prices': 'high', 'arena_capacity': 20478,
                'social_media_following': 1300000, 'bandwagon_factor': 0.3, 'local_competition': ['Eagles', 'Phillies'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['Kevin Hart', 'Meek Mill']
            },
            'PHX': {
                'city': 'Phoenix', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.65,
                'media_market_rank': 11, 'ticket_prices': 'medium', 'arena_capacity': 18055,
                'social_media_following': 750000, 'bandwagon_factor': 0.5, 'local_competition': ['Cardinals', 'Diamondbacks'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.75, 'celebrity_fans': ['Alice Cooper']
            },
            'POR': {
                'city': 'Portland', 'market_size': 'medium', 'recent_success': 'medium', 'fanbase_loyalty': 0.95,
                'media_market_rank': 22, 'ticket_prices': 'high', 'arena_capacity': 19393,
                'social_media_following': 850000, 'bandwagon_factor': 0.1, 'local_competition': ['Timbers'],
                'ownership_stability': 0.95, 'front_office_reputation': 0.8, 'celebrity_fans': ['Bill Walton']
            },
            'SAC': {
                'city': 'Sacramento', 'market_size': 'medium', 'recent_success': 'low', 'fanbase_loyalty': 0.85,
                'media_market_rank': 20, 'ticket_prices': 'medium', 'arena_capacity': 17608,
                'social_media_following': 500000, 'bandwagon_factor': 0.2, 'local_competition': ['49ers'],
                'ownership_stability': 0.8, 'front_office_reputation': 0.7, 'celebrity_fans': ['E-40']
            },
            'SAS': {
                'city': 'San Antonio', 'market_size': 'medium', 'recent_success': 'medium', 'fanbase_loyalty': 0.95,
                'media_market_rank': 31, 'ticket_prices': 'low', 'arena_capacity': 18418,
                'social_media_following': 900000, 'bandwagon_factor': 0.2, 'local_competition': ['None'],
                'ownership_stability': 0.95, 'front_office_reputation': 0.95, 'celebrity_fans': ['George Strait']
            },
            'TOR': {
                'city': 'Toronto', 'market_size': 'large', 'recent_success': 'medium', 'fanbase_loyalty': 0.8,
                'media_market_rank': 4, 'ticket_prices': 'high', 'arena_capacity': 19800,
                'social_media_following': 1200000, 'bandwagon_factor': 0.4, 'local_competition': ['Maple Leafs', 'Raptors'],
                'ownership_stability': 0.85, 'front_office_reputation': 0.8, 'celebrity_fans': ['Drake', 'The Weeknd']
            },
            'UTA': {
                'city': 'Utah', 'market_size': 'small', 'recent_success': 'medium', 'fanbase_loyalty': 0.95,
                'media_market_rank': 30, 'ticket_prices': 'medium', 'arena_capacity': 18306,
                'social_media_following': 600000, 'bandwagon_factor': 0.1, 'local_competition': ['Real Salt Lake'],
                'ownership_stability': 0.9, 'front_office_reputation': 0.8, 'celebrity_fans': ['Donny Osmond']
            },
            'WAS': {
                'city': 'Washington', 'market_size': 'large', 'recent_success': 'low', 'fanbase_loyalty': 0.4,
                'media_market_rank': 9, 'ticket_prices': 'medium', 'arena_capacity': 20356,
                'social_media_following': 700000, 'bandwagon_factor': 0.3, 'local_competition': ['Commanders', 'Nationals'],
                'ownership_stability': 0.6, 'front_office_reputation': 0.5, 'celebrity_fans': ['Kevin Durant']
            }
        }
        
        # NBA season calendar (realistic seasonality)
        self.season_periods = {
            'preseason': {'months': [9, 10], 'engagement_multiplier': 0.6},
            'regular_season': {'months': [10, 11, 12, 1, 2, 3, 4], 'engagement_multiplier': 1.0},
            'playoffs': {'months': [4, 5, 6], 'engagement_multiplier': 1.5},
            'offseason': {'months': [7, 8], 'engagement_multiplier': 0.3}
        }
        
        # Get segment configurations from config
        segments = synthetic_config['segments']
        
        # ================================================================
        # CUSTOMER SEGMENT CONFIGURATIONS - Psychological and behavioral profiles
        # ================================================================
        # Four distinct fan segments with realistic psychological factors:
        # - casual: Price-sensitive, high churn, social media driven
        # - regular: Balanced engagement, moderate loyalty 
        # - avid: Strong team loyalty, family-oriented, consistent spending
        # - super_fan: Premium tier, low churn, highest engagement
        self.segment_configs = {
            'casual': {
                'weight': segments.get('casual', 0.4),
                'plan_tiers': {'basic': 0.7, 'premium': 0.3},
                'price_range': (15, 35),
                'auto_renew_prob': 0.6,
                'churn_base_rate': 0.35,  # 35% annual churn for casual fans
                'engagement_mean': 0.3,
                'spend_multiplier': 0.8,
                'team_loyalty': 0.4,
                'seasonal_sensitivity': 0.8,
                'price_sensitivity': 0.7,
                'social_influence': 0.6,  # How much others' opinions matter
                'fomo_factor': 0.4,  # Fear of missing out
                'brand_loyalty': 0.3,  # Loyalty to NBA brand vs team
                'impulse_buying': 0.7,  # Tendency for impulse purchases
                'marketing_susceptibility': 0.8,  # Response to marketing
                'recession_sensitivity': 0.9,  # Economic impact
                'streaming_preference': 0.8,  # Prefers streaming over cable
                'mobile_first': 0.9,  # Mobile app usage preference
                'social_media_activity': 0.7  # Social media engagement level
            },
            'regular': {
                'weight': segments.get('regular', 0.3),
                'plan_tiers': {'basic': 0.2, 'premium': 0.6, 'vip': 0.2},
                'price_range': (25, 75),
                'auto_renew_prob': 0.9,
                'churn_base_rate': 0.18,  # 18% annual churn for regular fans
                'engagement_mean': 0.8,
                'spend_multiplier': 1.5,
                'team_loyalty': 0.7,
                'seasonal_sensitivity': 0.6,
                'price_sensitivity': 0.4,
                'social_influence': 0.4,
                'fomo_factor': 0.6,
                'brand_loyalty': 0.6,
                'impulse_buying': 0.5,
                'marketing_susceptibility': 0.6,
                'recession_sensitivity': 0.6,
                'streaming_preference': 0.6,
                'mobile_first': 0.7,
                'social_media_activity': 0.6
            },
            'avid': {
                'weight': segments.get('avid', 0.2),
                'plan_tiers': {'basic': 0.1, 'premium': 0.4, 'family': 0.5},
                'price_range': (35, 85),
                'auto_renew_prob': 0.85,
                'churn_base_rate': 0.25,  # 25% annual churn for avid fans
                'engagement_mean': 0.7,
                'spend_multiplier': 1.8,
                'team_loyalty': 0.8,
                'seasonal_sensitivity': 0.4,
                'price_sensitivity': 0.3,
                'social_influence': 0.3,
                'fomo_factor': 0.8,
                'brand_loyalty': 0.7,
                'impulse_buying': 0.6,
                'marketing_susceptibility': 0.5,
                'recession_sensitivity': 0.4,
                'streaming_preference': 0.5,
                'mobile_first': 0.6,
                'social_media_activity': 0.8
            },
            'super_fan': {
                'weight': segments.get('super_fan', 0.1),
                'plan_tiers': {'premium': 0.3, 'vip': 0.7},
                'price_range': (60, 200),
                'auto_renew_prob': 0.95,
                'churn_base_rate': 0.08,  # 8% annual churn for super fans
                'engagement_mean': 0.9,
                'spend_multiplier': 3.0,
                'team_loyalty': 0.95,
                'seasonal_sensitivity': 0.2,
                'price_sensitivity': 0.1,
                'social_influence': 0.2,
                'fomo_factor': 0.9,
                'brand_loyalty': 0.9,
                'impulse_buying': 0.8,
                'marketing_susceptibility': 0.3,
                'recession_sensitivity': 0.2,
                'streaming_preference': 0.4,
                'mobile_first': 0.5,
                'social_media_activity': 0.9
            }
        }
        
        # Ultra-realistic demographics with psychological and behavioral factors
        self.demographics = {
            'age_groups': {
                '18-24': {
                    'weight': 0.15, 'spending_multiplier': 0.7, 'tech_savvy': 0.95,
                    'social_media_hours': 4.2, 'attention_span': 0.3, 'brand_switching': 0.8,
                    'mobile_preference': 0.95, 'streaming_native': 0.9, 'peer_influence': 0.9,
                    'economic_awareness': 0.4, 'long_term_planning': 0.3, 'impulse_control': 0.4
                },
                '25-34': {
                    'weight': 0.25, 'spending_multiplier': 1.2, 'tech_savvy': 0.85,
                    'social_media_hours': 3.1, 'attention_span': 0.5, 'brand_switching': 0.6,
                    'mobile_preference': 0.9, 'streaming_native': 0.8, 'peer_influence': 0.7,
                    'economic_awareness': 0.7, 'long_term_planning': 0.6, 'impulse_control': 0.6
                },
                '35-44': {
                    'weight': 0.25, 'spending_multiplier': 1.5, 'tech_savvy': 0.7,
                    'social_media_hours': 2.3, 'attention_span': 0.7, 'brand_switching': 0.4,
                    'mobile_preference': 0.8, 'streaming_native': 0.6, 'peer_influence': 0.5,
                    'economic_awareness': 0.8, 'long_term_planning': 0.8, 'impulse_control': 0.7
                },
                '45-54': {
                    'weight': 0.20, 'spending_multiplier': 1.3, 'tech_savvy': 0.6,
                    'social_media_hours': 1.8, 'attention_span': 0.8, 'brand_switching': 0.3,
                    'mobile_preference': 0.7, 'streaming_native': 0.4, 'peer_influence': 0.4,
                    'economic_awareness': 0.9, 'long_term_planning': 0.9, 'impulse_control': 0.8
                },
                '55-64': {
                    'weight': 0.10, 'spending_multiplier': 1.1, 'tech_savvy': 0.4,
                    'social_media_hours': 1.2, 'attention_span': 0.9, 'brand_switching': 0.2,
                    'mobile_preference': 0.5, 'streaming_native': 0.3, 'peer_influence': 0.3,
                    'economic_awareness': 0.9, 'long_term_planning': 0.9, 'impulse_control': 0.9
                },
                '65+': {
                    'weight': 0.05, 'spending_multiplier': 0.8, 'tech_savvy': 0.25,
                    'social_media_hours': 0.8, 'attention_span': 0.95, 'brand_switching': 0.1,
                    'mobile_preference': 0.3, 'streaming_native': 0.2, 'peer_influence': 0.2,
                    'economic_awareness': 0.95, 'long_term_planning': 0.95, 'impulse_control': 0.95
                }
            },
            'regions': {
                'northeast': {
                    'weight': 0.2, 'price_tolerance': 1.2, 'weather_sensitivity': 0.8,
                    'commute_time': 45, 'income_level': 1.15, 'education_level': 0.85,
                    'sports_culture': 0.9, 'tech_adoption': 0.8, 'urban_density': 0.9
                },
                'southeast': {
                    'weight': 0.25, 'price_tolerance': 0.9, 'weather_sensitivity': 0.6,
                    'commute_time': 35, 'income_level': 0.95, 'education_level': 0.75,
                    'sports_culture': 0.85, 'tech_adoption': 0.7, 'urban_density': 0.6
                },
                'midwest': {
                    'weight': 0.25, 'price_tolerance': 1.0, 'weather_sensitivity': 0.9,
                    'commute_time': 30, 'income_level': 1.0, 'education_level': 0.8,
                    'sports_culture': 0.95, 'tech_adoption': 0.75, 'urban_density': 0.7
                },
                'southwest': {
                    'weight': 0.15, 'price_tolerance': 1.1, 'weather_sensitivity': 0.4,
                    'commute_time': 40, 'income_level': 1.05, 'education_level': 0.78,
                    'sports_culture': 0.7, 'tech_adoption': 0.85, 'urban_density': 0.75
                },
                'west': {
                    'weight': 0.15, 'price_tolerance': 1.3, 'weather_sensitivity': 0.5,
                    'commute_time': 50, 'income_level': 1.25, 'education_level': 0.9,
                    'sports_culture': 0.8, 'tech_adoption': 0.95, 'urban_density': 0.85
                }
            }
        }
        
        # NEW HYPER-REALISTIC FEATURES
        
        # Advanced NBA injury and drama timeline
        self.injury_timeline = {
            2021: [('LAL', 0.15, 'LeBron/AD injuries'), ('BRK', 0.12, 'Kyrie/Harden issues')],
            2022: [('LAC', 0.18, 'Kawhi injury'), ('NOP', 0.10, 'Zion debut hype')],
            2023: [('MEM', 0.14, 'Ja Morant suspension'), ('PHX', 0.09, 'KD trade boost')],
            2024: [('PHI', 0.13, 'Embiid health concerns'), ('NYK', 0.08, 'Brunson breakout')],
            2025: [('MIA', 0.11, 'Butler aging concerns'), ('BOS', 0.07, 'Championship defense pressure')]
        }
        
        # Draft lottery and trade deadline effects
        self.draft_events = {
            2021: {'lottery_teams': ['DET', 'HOU', 'CLE'], 'excitement_boost': 0.08},
            2022: {'lottery_teams': ['ORL', 'OKC', 'HOU'], 'excitement_boost': 0.06},
            2023: {'lottery_teams': ['SA', 'CHA', 'POR'], 'excitement_boost': 0.12, 'wemby_factor': 0.25},
            2024: {'lottery_teams': ['ATL', 'WAS', 'POR'], 'excitement_boost': 0.05},
            2025: {'lottery_teams': ['BRK', 'POR', 'WAS'], 'excitement_boost': 0.07, 'rebuild_narratives': True}
        }
        
        # International player popularity effects
        self.international_stars = {
            'MIL': {'player': 'Giannis', 'international_boost': 0.15, 'peak_years': [2021, 2022, 2023]},
            'DAL': {'player': 'Luka', 'international_boost': 0.12, 'peak_years': [2022, 2023, 2024]},
            'DEN': {'player': 'Jokic', 'international_boost': 0.10, 'peak_years': [2021, 2022, 2023]},
            'TOR': {'player': 'Siakam', 'international_boost': 0.08, 'peak_years': [2021, 2022]},
            'SA': {'player': 'Wemby', 'international_boost': 0.20, 'peak_years': [2024], 'rookie_hype': True}
        }
        
        # Fantasy sports and gambling integration timeline
        self.fantasy_gambling_adoption = {
            2021: 0.25,  # COVID sports betting boom
            2022: 0.35,  # DraftKings/FanDuel expansion
            2023: 0.45,  # Legalization in more states
            2024: 0.55,  # Mainstream adoption
            2025: 0.62   # AI and prop bet integration
        }
        
        # Social media influencer effects and viral moments
        self.influencer_moments = {
            2021: [('Lakers championship run hype', 0.08), ('Crypto NBA partnerships', 0.05)],
            2022: [('Warriors dynasty return', 0.10), ('Shaq/Chuck viral moments', 0.06)],
            2023: [('Bronny/LeBron storyline', 0.12), ('NBA on TikTok expansion', 0.08)],
            2024: [('Olympic year hype', 0.15), ('Next-gen star emergence', 0.07)]
        }
        
        # Real attendance constraints and capacity modeling
        self.arena_constraints = {
            'LAL': {'capacity': 20000, 'demand_multiplier': 2.5, 'celebrity_factor': 0.15},
            'GSW': {'capacity': 18064, 'demand_multiplier': 2.2, 'tech_money_factor': 0.12},
            'NYK': {'capacity': 20789, 'demand_multiplier': 2.0, 'tourist_factor': 0.10},
            'BOS': {'capacity': 19156, 'demand_multiplier': 1.8, 'history_factor': 0.08},
            'CHI': {'capacity': 20917, 'demand_multiplier': 1.6, 'market_size_factor': 0.06},
            'MIA': {'capacity': 19600, 'demand_multiplier': 1.7, 'lifestyle_factor': 0.09},
            'PHX': {'capacity': 18422, 'demand_multiplier': 1.4, 'weather_factor': 0.05},
            'DAL': {'capacity': 20000, 'demand_multiplier': 1.3, 'texas_factor': 0.04}
        }
        
        # Advanced pricing psychology tiers
        self.pricing_psychology = {
            'value_seekers': {'weight': 0.35, 'price_elasticity': 1.8, 'discount_response': 0.9},
            'quality_focused': {'weight': 0.25, 'price_elasticity': 0.8, 'discount_response': 0.3},
            'convenience_buyers': {'weight': 0.25, 'price_elasticity': 1.2, 'discount_response': 0.6},
            'status_conscious': {'weight': 0.15, 'price_elasticity': 0.4, 'discount_response': 0.2}
        }
        
        # Technology adoption curves by generation
        self.tech_adoption_curves = {
            '18-24': {'streaming': 0.95, 'mobile_first': 0.9, 'social_media': 0.95, 'crypto_aware': 0.4},
            '25-34': {'streaming': 0.85, 'mobile_first': 0.8, 'social_media': 0.8, 'crypto_aware': 0.3},
            '35-44': {'streaming': 0.65, 'mobile_first': 0.6, 'social_media': 0.6, 'crypto_aware': 0.15},
            '45-54': {'streaming': 0.45, 'mobile_first': 0.4, 'social_media': 0.4, 'crypto_aware': 0.05},
            '55-64': {'streaming': 0.30, 'mobile_first': 0.25, 'social_media': 0.25, 'crypto_aware': 0.02},
            '65+': {'streaming': 0.20, 'mobile_first': 0.15, 'social_media': 0.15, 'crypto_aware': 0.01}
        }

    # ================================================================
    # BEHAVIORAL MODELING HELPER FUNCTIONS
    # ================================================================
    # These functions calculate various realistic factors that influence
    # fan behavior: injuries, draft excitement, international stars,
    # fantasy sports, viral moments, technology adoption, etc.
    
    def _get_injury_impact(self, team: str, year: int) -> float:
        """
        Calculate injury/drama impact for a team in a given year.
        
        Args:
            team (str): NBA team code (e.g., 'LAL', 'BOS')
            year (int): Year to check for injury impacts
            
        Returns:
            float: Impact multiplier (1.0 = no impact, <1.0 = negative impact)
        """
        if year in self.injury_timeline:
            for team_code, impact, _ in self.injury_timeline[year]:
                if team == team_code:
                    return 1.0 - impact  # Convert to negative impact on engagement
        return 1.0  # No impact found
    
    def _get_draft_excitement(self, team: str, year: int) -> float:
        """
        Calculate draft lottery excitement boost for rebuilding teams.
        
        Args:
            team (str): NBA team code
            year (int): Draft year to check
            
        Returns:
            float: Engagement boost multiplier (1.0+ = positive boost)
        """
        if year in self.draft_events:
            draft_data = self.draft_events[year]
            if team in draft_data['lottery_teams']:
                base_boost = draft_data['excitement_boost']
                # Special case: Wembanyama hype for Spurs in 2023
                if year == 2023 and team == 'SAS':
                    wemby_factor = draft_data.get('wemby_factor', 0)
                    return 1.0 + base_boost + wemby_factor
                return 1.0 + base_boost
        return 1.0  # No draft excitement
    
    def _get_international_boost(self, team: str, year: int) -> float:
        """
        Calculate international player popularity boost.
        
        Teams with popular international stars (Giannis, Luka, Jokic) get
        engagement boosts, especially during their peak performance years.
        
        Args:
            team (str): NBA team code
            year (int): Year to check for international star impact
            
        Returns:
            float: Engagement boost multiplier
        """
        if team in self.international_stars:
            star_data = self.international_stars[team]
            if year in star_data['peak_years']:
                boost = star_data['international_boost']
                # Extra boost for rookie hype (e.g., Wemby's rookie year)
                if star_data.get('rookie_hype') and year == 2024:
                    boost *= 1.5
                return 1.0 + boost
        return 1.0  # No international star impact
    
    def _get_fantasy_gambling_effect(self, customer_data: Dict, year: int) -> float:
        """Calculate fantasy sports/gambling engagement effect."""
        if year not in self.fantasy_gambling_adoption:
            return 1.0
        
        adoption_rate = self.fantasy_gambling_adoption[year]
        age_group = customer_data.get('age_group', '35-44')
        
        # Younger demographics more likely to engage with fantasy/gambling
        age_multipliers = {
            '18-24': 1.5, '25-34': 1.3, '35-44': 1.0, 
            '45-54': 0.7, '55-64': 0.4, '65+': 0.2
        }
        
        age_mult = age_multipliers.get(age_group, 1.0)
        return 1.0 + (adoption_rate * age_mult * 0.3)  # Max 30% boost
    
    def _get_influencer_viral_boost(self, year: int, month: int) -> float:
        """Calculate viral moment/influencer boost."""
        if year in self.influencer_moments:
            # Viral moments happen randomly throughout the year
            moment_chance = np.random.random()
            if moment_chance < 0.1:  # 10% chance per month
                moments = self.influencer_moments[year]
                if moments:
                    _, boost = random.choice(moments)
                    return 1.0 + boost
        return 1.0
    
    def _get_capacity_constraint_effect(self, team: str, demand: float) -> float:
        """Calculate attendance constraint based on arena capacity."""
        if team in self.arena_constraints:
            constraint_data = self.arena_constraints[team]
            demand_mult = constraint_data['demand_multiplier']
            
            # Apply capacity constraints - if demand is too high, tickets become scarce
            if demand > demand_mult:
                scarcity_factor = min(demand / demand_mult, 3.0)  # Max 3x demand
                return 1.0 / scarcity_factor  # Reduced actual attendance due to sellouts
            
            # Apply special factors
            special_boost = 0
            for factor_key in ['celebrity_factor', 'tech_money_factor', 'tourist_factor', 
                             'history_factor', 'lifestyle_factor', 'weather_factor']:
                if factor_key in constraint_data:
                    special_boost += constraint_data[factor_key]
            
            return 1.0 + special_boost
        return 1.0
    
    def _get_pricing_psychology_effect(self, customer_data: Dict, price: float) -> float:
        """Calculate pricing psychology effect on purchase behavior."""
        # Assign customer to pricing psychology segment
        psychology_weights = [p['weight'] for p in self.pricing_psychology.values()]
        psychology_segment = np.random.choice(
            list(self.pricing_psychology.keys()), 
            p=psychology_weights
        )
        
        psychology_data = self.pricing_psychology[psychology_segment]
        elasticity = psychology_data['price_elasticity']
        
        # Calculate price sensitivity effect
        # Higher elasticity = more sensitive to price changes
        base_price = 50.0  # Reference price
        price_ratio = price / base_price
        
        # Price sensitivity formula: demand = base * (price_ratio ^ -elasticity)
        demand_multiplier = price_ratio ** (-elasticity)
        
        return max(0.1, min(3.0, demand_multiplier))  # Constrain to reasonable bounds
    
    def _get_tech_adoption_effect(self, customer_data: Dict, interaction_type: str) -> float:
        """Calculate technology adoption effect on behavior."""
        age_group = customer_data.get('age_group', '35-44')
        if age_group not in self.tech_adoption_curves:
            return 1.0
        
        tech_profile = self.tech_adoption_curves[age_group]
        
        # Apply technology adoption rates to different interaction types
        if interaction_type == 'streaming':
            return tech_profile['streaming']
        elif interaction_type == 'mobile_app':
            return tech_profile['mobile_first']
        elif interaction_type == 'social_media':
            return tech_profile['social_media']
        elif interaction_type == 'crypto_engagement':
            return tech_profile['crypto_aware']
        
        return 1.0

    def _get_economic_factors(self, month_date: datetime) -> Dict[str, float]:
        """Get economic factors for a specific month with interpolation."""
        month_key = month_date.strftime('%Y-%m')
        
        # Find the closest economic data points
        available_dates = sorted(self.economic_timeline.keys())
        
        if month_key in self.economic_timeline:
            return self.economic_timeline[month_key]
        
        # Interpolate between closest dates
        before_date = None
        after_date = None
        
        for date in available_dates:
            if date <= month_key:
                before_date = date
            elif date > month_key and after_date is None:
                after_date = date
                break
        
        if before_date and after_date:
            # Linear interpolation
            before_data = self.economic_timeline[before_date]
            after_data = self.economic_timeline[after_date]
            
            # Calculate interpolation weight
            total_months = (pd.to_datetime(after_date) - pd.to_datetime(before_date)).days / 30.44
            elapsed_months = (month_date - pd.to_datetime(before_date)).days / 30.44
            weight = elapsed_months / total_months if total_months > 0 else 0
            
            return {
                'recession_factor': before_data['recession_factor'] * (1 - weight) + after_data['recession_factor'] * weight,
                'inflation': before_data['inflation'] * (1 - weight) + after_data['inflation'] * weight,
                'unemployment': before_data['unemployment'] * (1 - weight) + after_data['unemployment'] * weight
            }
        elif before_date:
            return self.economic_timeline[before_date]
        elif after_date:
            return self.economic_timeline[after_date]
        else:
            # Default values
            return {'recession_factor': 1.0, 'inflation': 0.03, 'unemployment': 0.05}

    def _get_superstar_impact(self, team: str, month_date: datetime) -> float:
        """Calculate superstar player impact on team's fan engagement."""
        year = str(month_date.year)
        impact_multiplier = 1.0
        
        for player, data in self.superstar_timeline.items():
            if team in data['teams']:
                if year in data['peak_years']:
                    impact_multiplier *= 1.25  # 25% boost for superstar in peak
                elif year in data['decline_years']:
                    impact_multiplier *= 0.9   # 10% decline for aging superstar
                else:
                    impact_multiplier *= 1.1   # 10% boost for regular superstar
        
        return min(impact_multiplier, 1.8)  # Cap at 80% boost

    def _get_championship_effects(self, team: str, month_date: datetime) -> Dict[str, float]:
        """Get championship-related effects on fan behavior."""
        year = str(month_date.year)
        effects = {'winner_boost': 1.0, 'runner_up_boost': 1.0, 'cinderella_boost': 1.0, 'disappointment_penalty': 1.0}
        
        if year in self.championship_timeline:
            year_data = self.championship_timeline[year]
            
            if team == year_data['champion']:
                effects['winner_boost'] = 1.4  # 40% boost for champions
            elif team == year_data['runner_up']:
                effects['runner_up_boost'] = 1.2  # 20% boost for runner-up
            elif team in year_data.get('cinderella', []):
                effects['cinderella_boost'] = 1.3  # 30% boost for cinderella story
            elif team in year_data.get('disappointments', []):
                effects['disappointment_penalty'] = 0.8  # 20% penalty for disappointments
        
        return effects

    def _get_viral_moment_boost(self, month_date: datetime) -> float:
        """Get viral moment engagement boost for the month."""
        month_key = month_date.strftime('%Y-%m')
        if month_key in self.viral_moments:
            return 1.15  # 15% boost during viral moments
        return 1.0

    def _get_weather_impact(self, region: str, month: int) -> float:
        """Get weather impact on engagement and attendance."""
        region_weather = self.weather_patterns.get(region, {})
        
        if month in [12, 1, 2]:  # Winter
            return region_weather.get('winter_impact', 1.0)
        elif month in [6, 7, 8]:  # Summer
            return region_weather.get('summer_boost', 1.0)
        elif month in [6, 7, 8, 9] and region == 'southeast':  # Hurricane season
            return region_weather.get('hurricane_season', 1.0)
        elif month in [4, 5, 6] and region == 'midwest':  # Tornado season
            return region_weather.get('tornado_season', 1.0)
        elif month in [7, 8, 9, 10] and region == 'west':  # Wildfire season
            return region_weather.get('wildfire_season', 1.0)
        
        return 1.0

    def _get_streaming_preference_shift(self, year: int, age_group: str) -> float:
        """Model the shift from cable to streaming based on year and demographics."""
        base_streaming = self.streaming_landscape[str(year)]['streaming_adoption']
        age_info = self.demographics['age_groups'][age_group]
        
        # Younger people adopt streaming faster
        streaming_preference = base_streaming * (1 + age_info['streaming_native'] * 0.3)
        return min(streaming_preference, 0.9)  # Cap at 90%

    def _calculate_social_influence_effect(self, customer_data: Dict, team_performance: float, viral_boost: float) -> float:
        """Calculate how social influence affects individual customer behavior."""
        segment_config = self.segment_configs[customer_data['segment']]
        age_info = self.demographics['age_groups'][customer_data['age_group']]
        
        # Base social influence susceptibility
        social_susceptibility = segment_config['social_influence'] * age_info['peer_influence']
        
        # Good team performance creates positive social buzz
        social_buzz_factor = 0.8 + (team_performance * 0.4)  # Range: 0.8 to 1.2
        
        # Viral moments amplify social effects
        viral_amplification = viral_boost
        
        # Calculate final social influence multiplier
        social_effect = 1.0 + (social_susceptibility * (social_buzz_factor - 1.0) * viral_amplification)
        
        return np.clip(social_effect, 0.7, 1.5)  # Range: 0.7x to 1.5x

    def _calculate_realistic_churn_probability(self, customer_data: Dict, month_data: Dict) -> float:
        """
        Calculate ultra-realistic churn probability based on multiple behavioral factors.
        
        This function models the complex psychology of fan churn by considering:
        - Economic conditions (recession, inflation, unemployment)
        - Team performance and fan loyalty
        - Seasonal patterns and competition from other sports
        - Social influence and viral moments
        - Technology adoption frustration
        - Pricing psychology
        
        Args:
            customer_data (Dict): Customer profile including segment, demographics, preferences
            month_data (Dict): Monthly context including economics, team performance, seasonality
            
        Returns:
            float: Churn probability between 0 and 1
        """
        segment_config = self.segment_configs[customer_data['segment']]
        base_churn = segment_config['churn_base_rate']
        
        # Calculate individual factor multipliers
        time_multiplier = self._calculate_time_factor(month_data['months_since_signup'])
        economic_multiplier = self._calculate_economic_factor(month_data['economic_factors'], segment_config)
        team_multiplier = self._calculate_team_performance_factor(month_data['team_win_rate'], customer_data['team_loyalty_score'])
        seasonal_multiplier = self._calculate_seasonal_factor(month_data['season_multiplier'], segment_config)
        social_multiplier = self._calculate_social_factor(month_data['social_influence_effect'])
        price_multiplier = self._calculate_price_factor(customer_data, segment_config)
        competition_multiplier = self._calculate_competition_factor(month_data['calendar_month'])
        
        # Advanced realistic factors
        advanced_multipliers = self._calculate_advanced_churn_factors(customer_data, month_data)
        
        # Combine all factors
        total_multiplier = (
            time_multiplier * 
            economic_multiplier * 
            team_multiplier * 
            seasonal_multiplier * 
            social_multiplier * 
            price_multiplier * 
            competition_multiplier * 
            advanced_multipliers
        )
        
        # Calculate final churn probability
        churn_probability = base_churn * total_multiplier
        
        # Apply realistic bounds and noise
        churn_probability = np.clip(churn_probability, 0.001, 0.8)  # Min 0.1%, max 80%
        churn_probability += np.random.normal(0, 0.01)  # Small random noise
        
        return max(0.0, min(1.0, churn_probability))
    
    def _calculate_time_factor(self, months_since_signup: int) -> float:
        """Calculate time-based churn factor (gradual increase over time)."""
        return 1.0 + (months_since_signup * 0.005)
    
    def _calculate_economic_factor(self, economic_factors: Dict, segment_config: Dict) -> float:
        """Calculate economic impact on churn (recession, inflation, unemployment)."""
        recession_impact = 2.0 - economic_factors['recession_factor']
        inflation_impact = 1.0 + (economic_factors['inflation'] - 0.02) * 2
        unemployment_impact = 1.0 + (economic_factors['unemployment'] - 0.04) * 3
        
        economic_multiplier = recession_impact * inflation_impact * unemployment_impact
        return economic_multiplier * segment_config['recession_sensitivity']
    
    def _calculate_team_performance_factor(self, team_win_rate: float, loyalty_score: float) -> float:
        """Calculate team performance impact on churn (moderated by loyalty)."""
        team_factor = 2.5 - (team_win_rate * 2.0)  # Poor performance increases churn
        return team_factor ** loyalty_score  # Loyalty moderates impact
    
    def _calculate_seasonal_factor(self, season_multiplier: float, segment_config: Dict) -> float:
        """Calculate seasonal churn factor (offseason increases churn)."""
        season_factor = 2.5 - season_multiplier
        return season_factor * segment_config['seasonal_sensitivity']
    
    def _calculate_social_factor(self, social_influence_effect: float) -> float:
        """Calculate social influence impact on churn."""
        return 2.0 - social_influence_effect  # Negative social buzz increases churn
    
    def _calculate_price_factor(self, customer_data: Dict, segment_config: Dict) -> float:
        """Calculate price sensitivity impact on churn."""
        if customer_data.get('price_increase_this_year', False):
            return 1.0 + (segment_config['price_sensitivity'] * 0.5)
        return 1.0
    
    def _calculate_competition_factor(self, calendar_month: int) -> float:
        """Calculate competitive sports impact on churn."""
        if calendar_month in [9, 10, 11]:  # NFL season overlap
            return 1.2
        elif calendar_month in [3, 4]:  # March Madness
            return 1.1
        return 1.0
    
    def _calculate_advanced_churn_factors(self, customer_data: Dict, month_data: Dict) -> float:
        """Calculate advanced realistic churn factors (injuries, draft, viral moments, etc.)."""
        year = month_data['year']
        calendar_month = month_data['calendar_month']
        favorite_team = customer_data['favorite_team']
        
        # Team-specific factors
        injury_factor = self._get_injury_impact(favorite_team, year)
        draft_factor = 2.0 - self._get_draft_excitement(favorite_team, year)
        international_factor = 2.0 - self._get_international_boost(favorite_team, year)
        
        # Behavioral factors
        fantasy_factor = 2.0 - self._get_fantasy_gambling_effect(customer_data, year)
        viral_factor = 2.0 - self._get_influencer_viral_boost(year, calendar_month)
        
        # Technology frustration (affects older users)
        tech_frustration = self._calculate_tech_frustration(customer_data)
        
        return (
            injury_factor * 
            draft_factor * 
            international_factor * 
            fantasy_factor * 
            viral_factor * 
            tech_frustration
        )
    
    def _calculate_tech_frustration(self, customer_data: Dict) -> float:
        """Calculate technology adoption frustration factor for older users."""
        age_group = customer_data['age_group']
        if age_group in ['55-64', '65+']:
            streaming_adoption = self._get_tech_adoption_effect(customer_data, 'streaming')
            if streaming_adoption < 0.5:  # Low tech adoption creates frustration
                return 1.3
        return 1.0
        pricing_psychology_factor = self._get_pricing_psychology_effect(customer_data, customer_data['price'])
        pricing_psychology_factor = 2.0 - (pricing_psychology_factor - 1.0) * 0.5  # Convert to churn factor
        
        # Calculate final churn probability with ALL factors
        final_churn = (base_churn * time_factor * economic_multiplier * 
                      team_factor * season_factor * social_factor * 
                      price_factor * competition_factor * injury_factor *
                      draft_factor * international_factor * fantasy_factor *
                      viral_factor * tech_frustration * pricing_psychology_factor)
        
        # Apply age-based stability (older customers more stable)
        stability_factor = age_info['long_term_planning']
        final_churn *= (2.0 - stability_factor)
        
        # Apply brand loyalty factor
        brand_loyalty_factor = segment_config['brand_loyalty']
        final_churn *= (1.5 - brand_loyalty_factor * 0.5)
        
        return np.clip(final_churn, 0.001, 0.5)  # Realistic range: 0.1% to 50% monthly churn
    
    def _get_season_period(self, month: int) -> str:
        """Determine which season period a month falls into."""
        for period, config in self.season_periods.items():
            if month in config['months']:
                return period
        return 'offseason'

    def _get_engagement_multiplier(self, month: int) -> float:
        """Get engagement multiplier based on season."""
        period = self._get_season_period(month)
        return self.season_periods[period]['engagement_multiplier']

    def _get_economic_churn_impact(self, year: int, month: int) -> float:
        """Calculate economic impact on churn probability."""
        month_key = f'{year}-{month:02d}'
        
        # Find closest economic data
        economic_data = None
        for key, data in self.economic_timeline.items():
            if key <= month_key:
                economic_data = data
            else:
                break
        
        if not economic_data:
            return 1.0
        
        # Higher inflation and unemployment increase churn
        inflation_impact = 1 + (economic_data['inflation'] - 0.02) * 2.0  # Above 2% baseline increases churn
        unemployment_impact = 1 + (economic_data['unemployment'] - 0.04) * 3.0  # Above 4% baseline increases churn
        recession_impact = 2.0 - economic_data['recession_factor']  # Lower factor = higher churn
        
        return max(0.5, min(2.5, inflation_impact * unemployment_impact * recession_impact))
    
    def _get_pricing_psychology_impact(self, customer_data: Dict, price: float, segment_config: Dict) -> float:
        """Calculate pricing psychology impact on churn."""
        price_sensitivity = segment_config['price_sensitivity']
        
        # Compare to segment average price
        avg_price = (segment_config['price_range'][0] + segment_config['price_range'][1]) / 2
        price_ratio = price / avg_price
        
        # Higher prices relative to segment average increase churn
        if price_ratio > 1.2:  # 20% above average
            return 1.0 + ((price_ratio - 1.2) * price_sensitivity * 2.0)
        elif price_ratio < 0.8:  # 20% below average (less churn)
            return 1.0 - ((0.8 - price_ratio) * 0.3)
        
        return 1.0
    
    def _get_streaming_competition_churn(self, year: int) -> float:
        """Calculate churn impact from streaming competition."""
        if str(year) in self.streaming_landscape:
            streaming_data = self.streaming_landscape[str(year)]
            illegal_stream_rate = streaming_data['illegal_streams']
            # Higher illegal streaming availability increases churn
            return 1.0 + (illegal_stream_rate * 1.5)
        return 1.0

    def _assign_favorite_team(self, region: str) -> str:
        """Assign favorite team based on region with realistic preferences."""
        regional_teams = {
            'northeast': ['BOS', 'NYK', 'BRK', 'PHI'],
            'southeast': ['ATL', 'CHA', 'MIA', 'ORL', 'WAS'],
            'midwest': ['CHI', 'CLE', 'DET', 'IND', 'MIL', 'MIN'],
            'southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS'],
            'west': ['DEN', 'GSW', 'LAC', 'LAL', 'PHX', 'POR', 'SAC', 'UTA']
        }
        
        # 70% chance of regional team, 30% chance of any popular team
        if np.random.random() < 0.7 and region in regional_teams:
            return np.random.choice(regional_teams[region])
        else:
            # Popular teams get higher selection probability
            popular_teams = ['LAL', 'GSW', 'BOS', 'MIA', 'CHI']
            if np.random.random() < 0.6:
                return np.random.choice(popular_teams)
            else:
                return np.random.choice(list(self.nba_teams.keys()))

    # ================================================================
    # MAIN DATA GENERATION FUNCTIONS
    # ================================================================
    # These functions generate the three main datasets:
    # 1. Customer profiles with demographics and preferences
    # 2. Team performance data with realistic seasonality
    # 3. Customer interaction/engagement data with behavioral modeling

    def generate_customers(self) -> pd.DataFrame:
        """
        Generate enhanced synthetic customer data with demographics and team preferences.
        
        Creates realistic customer profiles including:
        - Customer segmentation (casual, regular, avid, super_fan)
        - Demographics (age, region, income level indicators)
        - Team preferences and loyalty scores
        - Behavioral characteristics (price sensitivity, tech adoption)
        - Psychological factors (brand loyalty, social influence susceptibility)
        
        Returns:
            pd.DataFrame: Customer dataset with comprehensive profiles
        """
        logger.info(f"Generating {self.num_customers} enhanced synthetic customers")
        
        customers = []
        
        for i in range(self.num_customers):
            # Assign segment based on configured weights
            segment = np.random.choice(
                list(self.segment_configs.keys()),
                p=[config['weight'] for config in self.segment_configs.values()]
            )
            segment_config = self.segment_configs[segment]
            
            # Assign demographics
            age_group = np.random.choice(
                list(self.demographics['age_groups'].keys()),
                p=[config['weight'] for config in self.demographics['age_groups'].values()]
            )
            
            region = np.random.choice(
                list(self.demographics['regions'].keys()),
                p=[config['weight'] for config in self.demographics['regions'].values()]
            )
            
            # Assign favorite team based on region
            favorite_team = self._assign_favorite_team(region)
            team_info = self.nba_teams[favorite_team]
            
            # Assign plan tier
            plan_tier = np.random.choice(
                list(segment_config['plan_tiers'].keys()),
                p=list(segment_config['plan_tiers'].values())
            )
            
            # Assign price with demographic and regional adjustments
            price_min, price_max = segment_config['price_range']
            base_price = np.random.uniform(price_min, price_max)
            
            # Apply demographic adjustments
            age_multiplier = self.demographics['age_groups'][age_group]['spending_multiplier']
            region_multiplier = self.demographics['regions'][region]['price_tolerance']
            
            price = base_price * age_multiplier * region_multiplier
            
            # Assign signup date (within date range)
            signup_days = (self.end_date - self.start_date).days
            signup_date = self.start_date + timedelta(days=np.random.randint(0, signup_days))
            
            # Auto-renew status influenced by team loyalty
            auto_renew_base = segment_config['auto_renew_prob']
            team_loyalty_boost = team_info['fanbase_loyalty'] * 0.1
            auto_renew_prob = min(0.95, auto_renew_base + team_loyalty_boost)
            auto_renew = np.random.random() < auto_renew_prob
            
            # Initial discount with some business logic
            if segment == 'super_fan':
                discount = np.random.choice([0, 0.1], p=[0.8, 0.2])  # Super fans get fewer discounts
            else:
                discount = np.random.choice([0, 0.1, 0.2, 0.3], p=[0.5, 0.25, 0.15, 0.1])
            
            # Fan characteristics
            years_following_team = max(1, int(np.random.gamma(2, 5)))  # How long they've been a fan
            
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'segment': segment,
                'plan_tier': plan_tier,
                'price': round(price, 2),
                'signup_date': signup_date.strftime('%Y-%m-%d'),
                'auto_renew': auto_renew,
                'initial_discount': discount,
                'age_group': age_group,
                'region': region,
                'favorite_team': favorite_team,
                'years_following_team': years_following_team,
                'team_loyalty_score': round(segment_config['team_loyalty'] * team_info['fanbase_loyalty'], 2)
            }
            
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        
        # Save to file with enhanced prefix
        output_file = self.data_paths['raw_synth'] / 'enhanced_customers.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced customer data to {output_file}")
        
        return df
    
    def generate_team_performance(self) -> pd.DataFrame:
        """Generate enhanced team performance data with realistic NBA patterns."""
        logger.info("Generating enhanced team performance data")
        
        # NBA team abbreviations
        teams = list(self.nba_teams.keys())
        
        team_data = []
        
        # Generate season-level team characteristics
        team_season_performance = {}
        
        for team in teams:
            team_info = self.nba_teams[team]
            
            # Assign team quality based on recent success
            if team_info['recent_success'] == 'high':
                base_quality = np.random.normal(0.7, 0.1)
            elif team_info['recent_success'] == 'medium':
                base_quality = np.random.normal(0.5, 0.15)
            else:
                base_quality = np.random.normal(0.3, 0.1)
            
            base_quality = np.clip(base_quality, 0.15, 0.85)
            
            # Market size affects resources and player acquisition
            market_multiplier = {'large': 1.1, 'medium': 1.0, 'small': 0.9}[team_info['market_size']]
            adjusted_quality = base_quality * market_multiplier
            
            team_season_performance[team] = {
                'base_quality': adjusted_quality,
                'injury_prone': np.random.random() < 0.3,  # 30% of teams are injury-prone
                'home_advantage': np.random.normal(0.05, 0.02)  # Home court advantage
            }
        
        # Generate monthly data
        for month_idx in range(self.num_months):
            month_date = self.start_date + timedelta(days=month_idx*30)
            calendar_month = month_date.month
            
            # Determine if it's a playoff month
            is_playoff_month = calendar_month in [4, 5, 6]
            
            for team in teams:
                team_perf = team_season_performance[team]
                
                # Base performance with monthly variation
                base_quality = team_perf['base_quality']
                monthly_variation = np.random.normal(0, 0.08)
                current_quality = np.clip(base_quality + monthly_variation, 0, 1)
                
                # Season-specific adjustments
                if is_playoff_month:
                    # Only good teams make playoffs
                    if base_quality > 0.5:
                        playoff_boost = np.random.normal(0.1, 0.05)
                        current_quality = min(0.95, current_quality + playoff_boost)
                    else:
                        # Bad teams don't play in playoffs
                        current_quality = 0
                
                # Win rate calculation
                win_rate = current_quality if not is_playoff_month or current_quality > 0 else 0
                
                # Point differential (more realistic NBA ranges)
                point_diff = (current_quality - 0.5) * 25 + np.random.normal(0, 3)
                
                # Games played varies by month and playoffs
                if calendar_month in [7, 8, 9]:  # Offseason
                    games_played = 0
                elif is_playoff_month:
                    if current_quality > 0:  # Team made playoffs
                        games_played = np.random.randint(4, 17)  # Playoff series length
                    else:
                        games_played = 0
                else:
                    games_played = np.random.randint(10, 16)  # Regular season
                
                # Attendance percentage (affected by team performance and market)
                base_attendance = 0.75 + (current_quality - 0.5) * 0.4
                market_boost = {'large': 0.05, 'medium': 0, 'small': -0.05}[self.nba_teams[team]['market_size']]
                attendance_pct = np.clip(base_attendance + market_boost + np.random.normal(0, 0.1), 0.3, 1.0)
                
                # Star player injuries (more realistic)
                star_player_games_missed = 0
                if team_perf['injury_prone'] and games_played > 0:
                    star_player_games_missed = np.random.poisson(1.5)
                    star_player_games_missed = min(star_player_games_missed, games_played)
                
                # Back-to-back games (realistic NBA scheduling)
                back_to_back_games = 0
                if games_played > 0 and not is_playoff_month:
                    back_to_back_games = min(np.random.poisson(2), games_played // 2)
                
                team_data.append({
                    'team': team,
                    'month': month_date.strftime('%Y-%m'),
                    'win_rate': round(win_rate, 3),
                    'avg_point_differential': round(point_diff, 1),
                    'games_played': games_played,
                    'attendance_percentage': round(attendance_pct, 3),
                    'back_to_back_games': back_to_back_games,
                    'star_player_games_missed': star_player_games_missed,
                    'is_playoff_month': is_playoff_month,
                    'home_advantage': round(team_perf['home_advantage'], 3)
                })
        
        df = pd.DataFrame(team_data)
        
        # Save to file with enhanced prefix
        output_file = self.data_paths['raw_synth'] / 'enhanced_team_performance.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced team performance data to {output_file}")
        
        return df
    
    def generate_interactions(self, customers_df: pd.DataFrame, 
                             team_performance_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate realistic customer interaction data with team performance correlations."""
        logger.info("Generating enhanced customer interaction data")
        
        interactions = []
        
        # Create team performance lookup for efficiency if provided
        team_perf_lookup = {}
        if team_performance_df is not None:
            for _, row in team_performance_df.iterrows():
                key = (row['team'], row['month'])
                team_perf_lookup[key] = row
        
        for _, customer in customers_df.iterrows():
            segment_config = self.segment_configs[customer['segment']]
            customer_id = customer['customer_id']
            signup_date = datetime.strptime(customer['signup_date'], '%Y-%m-%d')
            
            # Enhanced customer characteristics
            if 'favorite_team' in customer:
                favorite_team = customer['favorite_team']
                age_group = customer['age_group']
                age_info = self.demographics['age_groups'][age_group]
                team_loyalty = customer['team_loyalty_score']
                tech_savviness = age_info['tech_savvy']
            else:
                # Fallback for legacy data
                favorite_team = np.random.choice(list(self.nba_teams.keys()))
                age_info = {'spending_multiplier': 1.0, 'tech_savvy': 0.7}
                team_loyalty = 0.6
                tech_savviness = 0.7
            
            # Customer-specific characteristics
            base_engagement = np.random.beta(2, 3) * segment_config['engagement_mean'] + 0.1
            base_engagement = np.clip(base_engagement, 0.05, 0.95)
            
            spend_propensity = np.random.gamma(2, segment_config['spend_multiplier'])
            if 'age_group' in customer:
                spend_propensity *= age_info['spending_multiplier']
            
            # Track customer lifecycle
            is_active = True
            
            for month_idx in range(self.num_months):
                month_date = self.start_date + timedelta(days=month_idx*30)
                calendar_month = month_date.month
                
                # Skip months before signup
                if month_date < signup_date:
                    continue
                
                # Get team performance for this month
                team_perf_key = (favorite_team, month_date.strftime('%Y-%m'))
                team_perf = team_perf_lookup.get(team_perf_key, {})
                
                # Seasonal engagement multiplier
                season_multiplier = self._get_engagement_multiplier(calendar_month)
                
                # Team performance impact on engagement
                team_win_rate = team_perf.get('win_rate', 0.5)
                team_multiplier = 0.7 + (team_win_rate * 0.6)  # Range: 0.7 to 1.3
                
                # Loyalty affects how much team performance matters
                team_impact = team_loyalty * team_multiplier + (1 - team_loyalty) * 1.0
                
                # Base monthly engagement calculation
                monthly_engagement = base_engagement * season_multiplier * team_impact
                monthly_engagement = np.clip(monthly_engagement, 0.05, 1.0)
                
                # NEW HYPER-REALISTIC ENGAGEMENT FACTORS
                year = month_date.year
                
                # Apply injury/drama impact
                injury_multiplier = self._get_injury_impact(favorite_team, year)
                
                # Apply draft excitement
                draft_multiplier = self._get_draft_excitement(favorite_team, year)
                
                # Apply international star boost
                international_multiplier = self._get_international_boost(favorite_team, year)
                
                # Apply fantasy/gambling engagement boost
                fantasy_multiplier = self._get_fantasy_gambling_effect(customer.to_dict(), year)
                
                # Apply viral moment boost
                viral_multiplier = self._get_influencer_viral_boost(year, calendar_month)
                
                # Apply tech adoption effects
                streaming_adoption = self._get_tech_adoption_effect(customer.to_dict(), 'streaming')
                mobile_adoption = self._get_tech_adoption_effect(customer.to_dict(), 'mobile_app')
                social_adoption = self._get_tech_adoption_effect(customer.to_dict(), 'social_media')
                
                # Combine all new factors into engagement
                hyper_realistic_multiplier = (injury_multiplier * draft_multiplier * 
                                            international_multiplier * fantasy_multiplier * 
                                            viral_multiplier)
                
                # Apply to base engagement
                monthly_engagement *= hyper_realistic_multiplier
                monthly_engagement = np.clip(monthly_engagement, 0.05, 1.0)
                
                # Apply capacity constraints for ticket demand calculation (will be used later)
                base_ticket_demand = monthly_engagement * team_multiplier
                capacity_effect = self._get_capacity_constraint_effect(favorite_team, base_ticket_demand)
                
                # Calculate realistic churn probability
                months_since_signup = max(1, (month_date - signup_date).days // 30)
                
                # Convert annual churn rate to monthly (more realistic)
                annual_churn = segment_config['churn_base_rate']  # This is now treated as annual
                monthly_base_churn = 1 - (1 - annual_churn) ** (1/12)  # Convert to monthly
                
                # Churn factors (more realistic ranges)
                time_factor = 1 + (months_since_signup * 0.005)  # Gradual increase over time
                team_factor = 1.5 - (team_multiplier * 0.5)  # Poor team performance increases churn
                season_factor = 1.3 - (season_multiplier * 0.3)  # Offseason increases churn
                
                # Add economic and psychological factors
                economic_factor = self._get_economic_churn_impact(year, calendar_month)
                pricing_factor = self._get_pricing_psychology_impact(customer.to_dict(), 
                                                                   customer['price'], 
                                                                   segment_config)
                competition_factor = self._get_streaming_competition_churn(year)
                
                # Age and tenure effects
                age_churn_multiplier = 1.0
                if customer.get('age_group') == 'young':
                    age_churn_multiplier = 1.3  # Young customers churn more
                elif customer.get('age_group') == 'senior':
                    age_churn_multiplier = 0.7  # Seniors churn less
                
                # Calculate final churn probability with random variation
                churn_prob = (monthly_base_churn * time_factor * team_factor * 
                             season_factor * economic_factor * pricing_factor * 
                             competition_factor * age_churn_multiplier)
                
                # Add random noise to make it less predictable
                noise_factor = np.random.normal(1.0, 0.2)  # 20% random variation
                churn_prob *= max(0.3, min(1.7, noise_factor))  # Clamp noise
                
                # More realistic churn bounds
                churn_prob = np.clip(churn_prob, 0.0001, 0.15)  # Max 15% monthly churn
                
                # Apply churn decision
                if is_active and np.random.random() < churn_prob:
                    is_active = False
                
                # Reactivation logic - churned customers might come back
                if not is_active:
                    # Calculate reactivation probability based on segment and circumstances
                    months_churned = 1  # Simplified - just track this month
                    reactivation_base = segment_config.get('brand_loyalty', 0.3) * 0.02  # 2% base per month
                    
                    # Special events increase reactivation
                    reactivation_boost = 1.0
                    if team_perf.get('is_playoff_month', False):
                        reactivation_boost *= 3.0  # Playoffs bring people back
                    if team_win_rate > 0.7:  # Team doing well
                        reactivation_boost *= 2.0
                    if month_date.strftime('%Y-%m') in self.viral_moments:
                        reactivation_boost *= 1.5  # Viral moments bring attention
                    
                    reactivation_prob = reactivation_base * reactivation_boost
                    reactivation_prob = np.clip(reactivation_prob, 0, 0.1)  # Max 10% monthly reactivation
                    
                    if np.random.random() < reactivation_prob:
                        is_active = True  # Customer reactivates!
                
                if not is_active:
                    # Include churned customers with zero engagement
                    interactions.append({
                        'customer_id': customer_id,
                        'month': month_date.strftime('%Y-%m'),
                        'minutes_watched': 0,
                        'tickets_purchased': 0,
                        'games_attended': 0,
                        'merch_spend': 0.0,
                        'support_tickets': 0,
                        'app_logins': 0,
                        'social_media_interactions': 0,
                        'promo_exposure': 0,
                        'email_opens': 0,
                        'engagement_level': 0.0,
                        'team_win_rate': team_win_rate,
                        'is_playoff_month': team_perf.get('is_playoff_month', False),
                        'is_active': False
                    })
                    continue
                
                # Generate interactions based on engagement level for ACTIVE customers
                
                # Enhanced minutes watched with streaming adoption and viral effects
                base_minutes = monthly_engagement * 800 * streaming_adoption
                if team_perf.get('is_playoff_month', False) and team_win_rate > 0:
                    base_minutes *= 1.8  # Playoff boost
                # Apply viral moments boost to viewing
                base_minutes *= viral_multiplier
                # International players increase viewing
                base_minutes *= international_multiplier
                minutes_watched = max(0, int(np.random.gamma(2, base_minutes/2)))
                
                # Enhanced tickets purchased with hyper-realistic factors
                ticket_rate = monthly_engagement * 0.3 * capacity_effect
                if team_perf.get('is_playoff_month', False) and team_win_rate > 0:
                    ticket_rate *= 2.5  # Playoff ticket demand
                elif calendar_month in [7, 8]:  # Offseason
                    ticket_rate *= 0.1
                
                # Apply pricing psychology to ticket purchases
                pricing_effect = self._get_pricing_psychology_effect(customer.to_dict(), customer.get('price', 50))
                ticket_rate *= pricing_effect
                
                tickets_purchased = np.random.poisson(ticket_rate)
                
                # Games attended (for season ticket holders and high-engagement fans)
                games_attended = 0
                if tickets_purchased > 0 and monthly_engagement > 0.6:
                    games_attended = min(tickets_purchased, team_perf.get('games_played', 0))
                
                # Enhanced merchandise spend with international player effects
                base_merch_spend = spend_propensity * 25 * international_multiplier
                if calendar_month in [11, 12]:  # Holiday season
                    base_merch_spend *= 1.5
                if team_win_rate > 0.7:  # Winning team merchandise premium
                    base_merch_spend *= 1.3
                # Apply draft hype for merchandise
                base_merch_spend *= draft_multiplier
                merch_spend = max(0, np.random.exponential(base_merch_spend))
                
                # Support tickets (decrease with tech savviness, increase with age)
                support_base_rate = 0.15 * (1 - tech_savviness) * monthly_engagement
                # Tech frustration increases support tickets for older users
                if customer.get('age_group') in ['55-64', '65+'] and streaming_adoption < 0.5:
                    support_base_rate *= 1.5
                support_tickets = np.random.poisson(support_base_rate)
                
                # Enhanced app logins with mobile adoption
                app_login_rate = monthly_engagement * tech_savviness * mobile_adoption * 15
                app_logins = np.random.poisson(app_login_rate)
                
                # Enhanced social media engagement with viral effects
                social_base = monthly_engagement * social_adoption * team_multiplier * viral_multiplier * 5
                # Fantasy/gambling increases social engagement
                social_base *= fantasy_multiplier
                social_media_interactions = np.random.poisson(social_base)
                
                # Promo exposure (varies by segment and season)
                promo_rate = 1.0 + (1 - monthly_engagement) * 2  # More promos for less engaged
                if calendar_month in [9, 10]:  # Season start marketing push
                    promo_rate *= 1.5
                promo_exposure = np.random.poisson(promo_rate)
                
                # Email opens (correlated with engagement and age)
                email_open_rate = monthly_engagement * (0.5 + tech_savviness * 0.5) * 8
                email_opens = np.random.poisson(email_open_rate)
                
                interaction = {
                    'customer_id': customer_id,
                    'month': month_date.strftime('%Y-%m'),
                    'minutes_watched': minutes_watched,
                    'tickets_purchased': tickets_purchased,
                    'games_attended': games_attended,
                    'merch_spend': round(merch_spend, 2),
                    'support_tickets': support_tickets,
                    'app_logins': app_logins,
                    'social_media_interactions': social_media_interactions,
                    'promo_exposure': promo_exposure,
                    'email_opens': email_opens,
                    'engagement_level': round(monthly_engagement, 3),
                    'team_win_rate': team_perf.get('win_rate', 0.5),
                    'is_playoff_month': team_perf.get('is_playoff_month', False),
                    'is_active': is_active
                }
                
                interactions.append(interaction)
        
        df = pd.DataFrame(interactions)
        
        # Save to file with enhanced prefix
        output_file = self.data_paths['raw_synth'] / 'enhanced_customer_interactions.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced interaction data to {output_file}")
        
        return df
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all enhanced synthetic datasets."""
        logger.info("Starting enhanced synthetic data generation")
        
        # Generate customers first
        customers_df = self.generate_customers()
        
        # Generate team performance
        team_performance_df = self.generate_team_performance()
        
        # Generate customer interactions (using team performance data)
        interactions_df = self.generate_interactions(customers_df, team_performance_df)
        
        data = {
            'customers': customers_df,
            'team_performance': team_performance_df,
            'interactions': interactions_df
        }
        
        # Generate enhanced summary statistics
        self._generate_summary_stats(data)
        
        logger.info("Enhanced synthetic data generation completed")
        
        return data
    
    def _generate_summary_stats(self, data: Dict[str, pd.DataFrame]) -> None:
        """Generate and save enhanced summary statistics."""
        customers_df = data['customers']
        interactions_df = data['interactions']
        team_performance_df = data['team_performance']
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'data_enhancement': 'v2.0 - Basketball-specific realism enhancements',
            'num_customers': len(customers_df),
            'num_months': self.num_months,
            'date_range': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            
            # Customer demographics (enhanced if available)
            'customer_segments': customers_df['segment'].value_counts().to_dict(),
            'plan_tiers': customers_df['plan_tier'].value_counts().to_dict(),
            'avg_price': float(customers_df['price'].mean()),
            
            # Enhanced demographics (if available)
            **({
                'age_groups': customers_df['age_group'].value_counts().to_dict(),
                'regions': customers_df['region'].value_counts().to_dict(),
                'favorite_teams': customers_df['favorite_team'].value_counts().head(10).to_dict(),
                'team_loyalty_avg': float(customers_df['team_loyalty_score'].mean()),
            } if 'age_group' in customers_df.columns else {}),
            
            # Engagement metrics
            'total_interactions': len(interactions_df),
            
            # Enhanced metrics (if available)
            **({
                'avg_monthly_engagement': float(interactions_df['engagement_level'].mean()),
                'playoff_month_interactions': len(interactions_df[interactions_df['is_playoff_month'] == True]),
                'total_merch_spend': float(interactions_df['merch_spend'].sum()),
                'avg_monthly_merch_spend': float(interactions_df['merch_spend'].mean()),
                'avg_minutes_watched': float(interactions_df['minutes_watched'].mean()),
                'avg_app_logins': float(interactions_df['app_logins'].mean()),
                'avg_tickets_purchased': float(interactions_df['tickets_purchased'].mean()),
            } if 'engagement_level' in interactions_df.columns else {
                'avg_minutes_watched': float(interactions_df['minutes_watched'].mean()),
                'avg_tickets_purchased': float(interactions_df['tickets_purchased'].mean()),
                'total_merch_spend': float(interactions_df['merch_spend'].sum()),
            }),
            
            # Retention metrics
            'final_month': interactions_df['month'].max(),
            'active_customers_final_month': int(interactions_df[
                interactions_df['month'] == interactions_df['month'].max()
            ]['is_active'].sum()),
            'overall_retention_rate': float(interactions_df[
                interactions_df['month'] == interactions_df['month'].max()
            ]['is_active'].mean()),
            
            # Team performance insights (if available)
            **({
                'avg_team_win_rate': float(team_performance_df['win_rate'].mean()),
                'playoff_teams_count': len(team_performance_df[
                    (team_performance_df['is_playoff_month'] == True) & 
                    (team_performance_df['win_rate'] > 0)
                ]['team'].unique()) if 'is_playoff_month' in team_performance_df.columns else 0,
            } if 'win_rate' in team_performance_df.columns else {})
        }
        
        # Save summary
        summary_file = self.data_paths['raw_synth'] / 'enhanced_summary_stats.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved enhanced summary statistics to {summary_file}")


# Alias for backward compatibility
SyntheticDataGenerator = UltraRealisticSyntheticDataGenerator


def main():
    """Main function for standalone execution."""
    generator = UltraRealisticSyntheticDataGenerator()
    generator.generate_all_data()


if __name__ == "__main__":
    main()
