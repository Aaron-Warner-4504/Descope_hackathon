from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime
from langchain_core.messages import HumanMessage
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from descope import DescopeClient
from langchain_tavily import TavilySearch
from notion_client import Client
from langchain.memory import ConversationBufferMemory
import subprocess
import time
import psycopg2
import replicate
import re 

import logging
import json
import time
import random
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from playwright.sync_api import sync_playwright

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import tool

chat = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",  # correct model name on Groq
    groq_api_key=os.getenv("GROQ_API_KEY")  # make sure to set this securely
)

DB_CONFIG = {
    "host": "localhost",
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": 5432
}
#Content Creation
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
# Initialize Descope client
DESCOPE_PROJECT_ID = os.getenv("DESCOPE_PROJECT_ID")
DESCOPE_MANAGEMENT_KEY = os.getenv("DESCOPE_MANAGEMENT_KEY")

descope_client = DescopeClient(project_id=DESCOPE_PROJECT_ID, management_key=DESCOPE_MANAGEMENT_KEY)

#!/usr/bin/env python3
"""
AI-Powered Mental Health Assistant - Complete Application
========================================================

A comprehensive mental health assistant that integrates with various APIs
to provide personalized wellness support, mood tracking, and team wellness features.
Uses LangChain with Groq for intelligent decision-making and analysis.

Requirements:
- pip install langchain langchain-groq google-api-python-client notion-client slack-sdk spotipy requests python-dotenv google-auth-oauthlib google-auth-httplib2

Environment Variables Required:
- GROQ_API_KEY (Required)
- GOOGLE_CALENDAR_CREDENTIALS (Optional - path to JSON file)
- NOTION_TOKEN (Optional)
- SLACK_BOT_TOKEN (Optional)
- SPOTIFY_CLIENT_ID (Optional)
- SPOTIFY_CLIENT_SECRET (Optional)
- DISCORD_BOT_TOKEN (Optional)
- NOTION_MOOD_DATABASE_ID (Optional)
- NOTION_WEEKLY_REPORTS_PAGE_ID (Optional)

Usage:
    python mental_health_assistant.py

Author: AI Mental Health Assistant Team
Version: 2.0 - AI-Powered Edition
"""

import os
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pickle
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path
chat = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",  # correct model name on Groq
    groq_api_key=os.getenv("GROQ_API_KEY")  # make sure to set this securely
)
# Check and install dependencies
def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('langchain', 'langchain'),
        ('langchain_groq', 'langchain-groq'), 
        ('googleapiclient', 'google-api-python-client'),
        ('notion_client', 'notion-client'),
        ('slack_sdk', 'slack-sdk'),
        ('spotipy', 'spotipy'),
        ('requests', 'requests'),
        ('dotenv', 'python-dotenv'),
        ('google_auth_oauthlib', 'google-auth-oauthlib'),
        ('google.auth.transport.requests', 'google-auth-httplib2')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

# Third-party imports
try:
    from dotenv import load_dotenv
    from langchain.tools import tool
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, AIMessage
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the required packages and try again.")
    sys.exit(1)

# API clients (with graceful degradation)
try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logging.warning("Google API client not installed. Google Calendar features will be disabled.")

try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    logging.warning("Notion client not installed. Notion features will be disabled.")

try:
    from slack_sdk import WebClient as SlackClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logging.warning("Slack SDK not installed. Slack features will be disabled.")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    logging.warning("Spotipy not installed. Spotify features will be disabled.")

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logging.warning("Discord.py not installed. Discord features will be disabled.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mental_health_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data models for persistence
@dataclass
class UserMood:
    """Represents a user's mood entry"""
    timestamp: datetime
    mood_score: int  # 1-10 scale
    notes: str = ""
    stress_level: int = 5  # 1-10 scale

@dataclass
class WellnessBreak:
    """Represents a wellness break activity"""
    timestamp: datetime
    break_type: str
    duration_minutes: int
    completed: bool = False
    effectiveness_rating: int = 5  # 1-10 scale

@dataclass
class UserProfile:
    """Stores user preferences and learning data"""
    preferred_break_types: List[str]
    break_effectiveness: Dict[str, float]  # break_type -> average effectiveness
    mood_patterns: Dict[str, List[int]]  # day_of_week -> mood scores
    last_updated: datetime

class DataManager:
    """Handles persistence of user data and learning"""
    
    def __init__(self, data_dir: str = "mental_health_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.mood_file = self.data_dir / "moods.pkl"
        self.breaks_file = self.data_dir / "breaks.pkl"
        self.profile_file = self.data_dir / "profile.pkl"
        
        # Load existing data
        self.moods = self._load_data(self.mood_file, [])
        self.breaks = self._load_data(self.breaks_file, [])
        self.profile = self._load_data(self.profile_file, self._create_default_profile())
    
    def _load_data(self, file_path: Path, default):
        """Load pickled data or return default"""
        try:
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        return default
    
    def _save_data(self, data, file_path: Path):
        """Save data to pickle file"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    def _create_default_profile(self) -> UserProfile:
        """Create default user profile"""
        return UserProfile(
            preferred_break_types=["meditation", "walk", "stretching", "breathing"],
            break_effectiveness={},
            mood_patterns={},
            last_updated=datetime.now()
        )
    
    def add_mood(self, mood: UserMood):
        """Add mood entry and update patterns"""
        self.moods.append(mood)
        self._update_mood_patterns(mood)
        self._save_data(self.moods, self.mood_file)
        self._save_data(self.profile, self.profile_file)
    
    def add_break(self, break_entry: WellnessBreak):
        """Add break entry and update effectiveness"""
        self.breaks.append(break_entry)
        if break_entry.completed:
            self._update_break_effectiveness(break_entry)
        self._save_data(self.breaks, self.breaks_file)
        self._save_data(self.profile, self.profile_file)
    
    def _update_mood_patterns(self, mood: UserMood):
        """Update mood patterns by day of week"""
        day_name = mood.timestamp.strftime("%A")
        if day_name not in self.profile.mood_patterns:
            self.profile.mood_patterns[day_name] = []
        self.profile.mood_patterns[day_name].append(mood.mood_score)
        self.profile.last_updated = datetime.now()
    
    def _update_break_effectiveness(self, break_entry: WellnessBreak):
        """Update break effectiveness based on user rating"""
        break_type = break_entry.break_type
        if break_type not in self.profile.break_effectiveness:
            self.profile.break_effectiveness[break_type] = break_entry.effectiveness_rating
        else:
            # Running average
            current = self.profile.break_effectiveness[break_type]
            self.profile.break_effectiveness[break_type] = (current + break_entry.effectiveness_rating) / 2
        self.profile.last_updated = datetime.now()
    
    def get_recent_moods(self, days: int = 7) -> List[UserMood]:
        """Get moods from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [mood for mood in self.moods if mood.timestamp >= cutoff]
    
    def get_recent_breaks(self, days: int = 7) -> List[WellnessBreak]:
        """Get breaks from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [break_entry for break_entry in self.breaks if break_entry.timestamp >= cutoff]
    
    def get_best_break_types(self) -> List[str]:
        """Get break types ranked by effectiveness"""
        if not self.profile.break_effectiveness:
            return self.profile.preferred_break_types
        
        sorted_breaks = sorted(
            self.profile.break_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [break_type for break_type, _ in sorted_breaks]

# Initialize data manager
data_manager = DataManager()

# API Service Classes
class GoogleCalendarService:
    """Enhanced Google Calendar integration service with advanced analysis"""
    
    def __init__(self):
        self.service = None
        if GOOGLE_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Google Calendar service"""
        try:
            creds_path = os.getenv('GOOGLE_CALENDAR_CREDENTIALS')
            if not creds_path or not Path(creds_path).exists():
                logger.warning("Google Calendar credentials not found")
                return
            
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            creds = None
            token_path = 'token.json'
            
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                    creds = flow.run_local_server(port=8080)
                
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('calendar', 'v3', credentials=creds)
            logger.info("Google Calendar service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar: {e}")
    
    def get_events_in_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get calendar events in a specific time range"""
        if not self.service:
            return []
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time.isoformat() + 'Z',
                timeMax=end_time.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return events
            
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """Get upcoming calendar events"""
        if not self.service:
            return []
        
        try:
            now = datetime.utcnow()
            until = now + timedelta(hours=hours)
            return self.get_events_in_range(now, until)
            
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []
    
    def analyze_calendar_patterns(self, days_back: int = 30) -> Dict:
        """Analyze calendar patterns for stress and workload insights"""
        if not self.service:
            return {}
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            events = self.get_events_in_range(start_time, end_time)
            
            analysis = {
                "total_events": len(events),
                "stress_indicators": self._analyze_stress_patterns(events),
                "workload_by_day": self._analyze_workload_by_day(events),
                "meeting_frequency": self._analyze_meeting_frequency(events),
                "free_time_analysis": self._analyze_free_time(events),
                "peak_stress_times": self._identify_peak_stress_times(events),
                "recommendations": []
            }
            
            # Generate recommendations based on analysis
            analysis["recommendations"] = self._generate_calendar_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing calendar patterns: {e}")
            return {}
    
    def _analyze_stress_patterns(self, events: List[Dict]) -> Dict:
        """Analyze events for stress-inducing patterns"""
        stress_keywords = [
            'deadline', 'review', 'interview', 'presentation', 'demo',
            'urgent', 'critical', 'important', 'crisis', 'emergency',
            'board', 'executive', 'client', 'customer', 'stakeholder',
            'performance', 'evaluation', 'assessment', 'audit'
        ]
        
        stress_events = []
        back_to_back_count = 0
        long_meetings = []
        
        for i, event in enumerate(events):
            summary = event.get('summary', '').lower()
            
            # Check for stress keywords
            stress_score = sum(1 for keyword in stress_keywords if keyword in summary)
            if stress_score > 0:
                stress_events.append({
                    'title': event.get('summary'),
                    'stress_score': stress_score,
                    'start': event.get('start', {}).get('dateTime'),
                    'keywords_found': [kw for kw in stress_keywords if kw in summary]
                })
            
            # Check for back-to-back meetings
            if i > 0:
                prev_end = events[i-1].get('end', {}).get('dateTime')
                curr_start = event.get('start', {}).get('dateTime')
                
                if prev_end and curr_start:
                    # Parse times and check if they're within 15 minutes
                    try:
                        prev_end_dt = datetime.fromisoformat(prev_end.replace('Z', '+00:00'))
                        curr_start_dt = datetime.fromisoformat(curr_start.replace('Z', '+00:00'))
                        
                        if (curr_start_dt - prev_end_dt).total_seconds() <= 900:  # 15 minutes
                            back_to_back_count += 1
                    except:
                        pass
            
            # Check for long meetings (over 2 hours)
            start_time = event.get('start', {}).get('dateTime')
            end_time = event.get('end', {}).get('dateTime')
            
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds() / 3600  # hours
                    
                    if duration >= 2:
                        long_meetings.append({
                            'title': event.get('summary'),
                            'duration_hours': duration,
                            'start': start_time
                        })
                except:
                    pass
        
        return {
            'high_stress_events': stress_events,
            'back_to_back_meetings': back_to_back_count,
            'long_meetings': long_meetings,
            'total_stress_score': sum(event['stress_score'] for event in stress_events)
        }
    
    def _analyze_workload_by_day(self, events: List[Dict]) -> Dict:
        """Analyze workload distribution by day of week"""
        day_counts = {}
        day_hours = {}
        
        for event in events:
            start_time = event.get('start', {}).get('dateTime')
            end_time = event.get('end', {}).get('dateTime')
            
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    
                    day_name = start_dt.strftime('%A')
                    duration = (end_dt - start_dt).total_seconds() / 3600
                    
                    day_counts[day_name] = day_counts.get(day_name, 0) + 1
                    day_hours[day_name] = day_hours.get(day_name, 0) + duration
                    
                except:
                    pass
        
        return {
            'meetings_per_day': day_counts,
            'hours_per_day': day_hours,
            'busiest_day': max(day_counts.items(), key=lambda x: x[1]) if day_counts else None,
            'longest_day': max(day_hours.items(), key=lambda x: x[1]) if day_hours else None
        }
    
    def _analyze_meeting_frequency(self, events: List[Dict]) -> Dict:
        """Analyze meeting frequency and density"""
        if not events:
            return {}
        
        # Group events by date
        events_by_date = {}
        for event in events:
            start_time = event.get('start', {}).get('dateTime')
            if start_time:
                try:
                    date = datetime.fromisoformat(start_time.replace('Z', '+00:00')).date()
                    if date not in events_by_date:
                        events_by_date[date] = []
                    events_by_date[date].append(event)
                except:
                    pass
        
        daily_counts = [len(events) for events in events_by_date.values()]
        
        return {
            'average_meetings_per_day': sum(daily_counts) / len(daily_counts) if daily_counts else 0,
            'max_meetings_in_day': max(daily_counts) if daily_counts else 0,
            'days_with_no_meetings': len([count for count in daily_counts if count == 0]),
            'busy_days': len([count for count in daily_counts if count >= 5])
        }
    
    def _analyze_free_time(self, events: List[Dict]) -> Dict:
        """Analyze free time between meetings"""
        if len(events) < 2:
            return {"gaps": [], "average_gap": 0}
        
        gaps = []
        for i in range(1, len(events)):
            prev_end = events[i-1].get('end', {}).get('dateTime')
            curr_start = events[i].get('start', {}).get('dateTime')
            
            if prev_end and curr_start:
                try:
                    prev_end_dt = datetime.fromisoformat(prev_end.replace('Z', '+00:00'))
                    curr_start_dt = datetime.fromisoformat(curr_start.replace('Z', '+00:00'))
                    
                    gap_minutes = (curr_start_dt - prev_end_dt).total_seconds() / 60
                    
                    if 0 < gap_minutes < 480:  # Between 0 and 8 hours (same day)
                        gaps.append(gap_minutes)
                except:
                    pass
        
        return {
            "gaps": gaps,
            "average_gap": sum(gaps) / len(gaps) if gaps else 0,
            "short_gaps": len([gap for gap in gaps if gap < 30]),  # Less than 30 min
            "adequate_gaps": len([gap for gap in gaps if 30 <= gap <= 60])  # 30-60 min
        }
    
    def _identify_peak_stress_times(self, events: List[Dict]) -> Dict:
        """Identify times of day and days of week with highest stress"""
        stress_by_hour = {}
        stress_by_day = {}
        
        stress_keywords = ['deadline', 'review', 'interview', 'presentation', 'demo', 'urgent', 'critical']
        
        for event in events:
            summary = event.get('summary', '').lower()
            stress_score = sum(1 for keyword in stress_keywords if keyword in summary)
            
            if stress_score > 0:
                start_time = event.get('start', {}).get('dateTime')
                if start_time:
                    try:
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        hour = dt.hour
                        day = dt.strftime('%A')
                        
                        stress_by_hour[hour] = stress_by_hour.get(hour, 0) + stress_score
                        stress_by_day[day] = stress_by_day.get(day, 0) + stress_score
                    except:
                        pass
        
        return {
            'peak_stress_hours': sorted(stress_by_hour.items(), key=lambda x: x[1], reverse=True)[:3],
            'peak_stress_days': sorted(stress_by_day.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _generate_calendar_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on calendar analysis"""
        recommendations = []
        
        # Back-to-back meetings
        if analysis['stress_indicators']['back_to_back_meetings'] > 5:
            recommendations.append("🕐 You have many back-to-back meetings. Try scheduling 15-minute buffers between meetings for mental breaks.")
        
        # Long meetings
        long_meetings = analysis['stress_indicators']['long_meetings']
        if len(long_meetings) > 2:
            recommendations.append("⏰ Consider breaking long meetings into shorter sessions with breaks to maintain focus and energy.")
        
        # High stress events
        if analysis['stress_indicators']['total_stress_score'] > 10:
            recommendations.append("⚡ High-stress events detected. Schedule wellness breaks before and after important meetings.")
        
        return recommendations
    
    def find_optimal_break_slots(self, date: datetime = None, duration_minutes: int = 15) -> List[Dict]:
        """Find optimal time slots for wellness breaks"""
        if not self.service:
            return []
        
        if date is None:
            date = datetime.now().date()
        
        # Get events for the day
        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())
        
        events = self.get_events_in_range(start_time, end_time)
        
        # Find gaps between meetings
        optimal_slots = []
        work_start = start_time.replace(hour=9)  # Assume 9 AM start
        work_end = start_time.replace(hour=17)   # Assume 5 PM end
        
        # Sort events by start time
        events.sort(key=lambda x: x.get('start', {}).get('dateTime', ''))
        
        current_time = work_start
        
        for event in events:
            event_start = event.get('start', {}).get('dateTime')
            if event_start:
                try:
                    event_start_dt = datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                    
                    # If there's a gap, suggest a break slot
                    gap_minutes = (event_start_dt - current_time).total_seconds() / 60
                    
                    if gap_minutes >= duration_minutes + 10:  # At least 10 min buffer
                        optimal_slots.append({
                            'start_time': current_time,
                            'end_time': current_time + timedelta(minutes=duration_minutes),
                            'gap_available': gap_minutes,
                            'reason': f"Gap before {event.get('summary', 'meeting')}"
                        })
                    
                    # Update current time to end of this event
                    event_end = event.get('end', {}).get('dateTime')
                    if event_end:
                        current_time = datetime.fromisoformat(event_end.replace('Z', '+00:00'))
                        
                except:
                    pass
        
        # Add end-of-day slot if there's time
        if current_time < work_end:
            gap_minutes = (work_end - current_time).total_seconds() / 60
            if gap_minutes >= duration_minutes:
                optimal_slots.append({
                    'start_time': current_time,
                    'end_time': current_time + timedelta(minutes=duration_minutes),
                    'gap_available': gap_minutes,
                    'reason': "End of workday wind-down"
                })
        
        return optimal_slots[:5]  # Return top 5 slots
    
    def insert_break_event(self, start_time: datetime, duration_minutes: int, break_type: str) -> bool:
        """Insert a wellness break into calendar"""
        if not self.service:
            return False
        
        try:
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            event = {
                'summary': f'Wellness Break - {break_type.title()}',
                'description': f'Personalized wellness break: {break_type}',
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 5},
                    ],
                },
            }
            
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Wellness break scheduled: {event.get('htmlLink')}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling break: {e}")
            return False

class NotionService:
    """Enhanced Notion integration service with reading capabilities"""
    
    def __init__(self):
        self.client = None
        if NOTION_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Notion client"""
        try:
            token = os.getenv('NOTION_TOKEN')
            if not token:
                logger.warning("Notion token not provided")
                return
            
            self.client = NotionClient(auth=token)
            logger.info("Notion client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Notion: {e}")
    
    def read_database(self, database_id: str, filter_conditions: Dict = None, sorts: List = None) -> List[Dict]:
        """Read entries from a Notion database with optional filtering and sorting"""
        if not self.client:
            return []
        
        try:
            query_params = {
                "database_id": database_id,
                "page_size": 100
            }
            
            if filter_conditions:
                query_params["filter"] = filter_conditions
            
            if sorts:
                query_params["sorts"] = sorts
            
            response = self.client.databases.query(**query_params)
            
            # Extract and clean the data
            entries = []
            for page in response.get("results", []):
                entry = self._extract_page_properties(page)
                entries.append(entry)
            
            logger.info(f"Read {len(entries)} entries from Notion database")
            return entries
            
        except Exception as e:
            logger.error(f"Error reading Notion database: {e}")
            return []
    
    def search_notion_content(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search across Notion workspace for specific content"""
        if not self.client:
            return []
        
        try:
            response = self.client.search(
                query=query,
                page_size=max_results,
                filter={"property": "object", "value": "page"}
            )
            
            results = []
            for page in response.get("results", []):
                page_data = self._extract_page_properties(page)
                results.append(page_data)
            
            logger.info(f"Found {len(results)} Notion pages matching '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Notion: {e}")
            return []
    
    def get_tasks_and_projects(self) -> Dict:
        """Read tasks and projects from common Notion database patterns"""
        tasks_data = {
            "tasks": [],
            "projects": [],
            "goals": [],
            "notes": []
        }
        
        try:
            # Try to find common database names
            database_names = ["Tasks", "Projects", "Goals", "Notes", "To-Do", "Work"]
            
            for db_name in database_names:
                search_results = self.search_notion_content(db_name, max_results=5)
                
                for result in search_results:
                    if "database" in result.get("url", "").lower():
                        # This is likely a database, try to read it
                        db_id = result["id"]
                        entries = self.read_database(db_id)
                        
                        # Categorize based on content
                        if any(word in db_name.lower() for word in ["task", "todo", "to-do"]):
                            tasks_data["tasks"].extend(self._categorize_as_tasks(entries))
                        elif "project" in db_name.lower():
                            tasks_data["projects"].extend(self._categorize_as_projects(entries))
                        elif "goal" in db_name.lower():
                            tasks_data["goals"].extend(entries)
                        else:
                            tasks_data["notes"].extend(entries)
                        break
            
            return tasks_data
            
        except Exception as e:
            logger.error(f"Error getting tasks and projects: {e}")
            return tasks_data
    
    def _extract_page_properties(self, page: Dict) -> Dict:
        """Extract properties from a Notion page object"""
        try:
            properties = page.get("properties", {})
            extracted = {
                "id": page.get("id"),
                "title": self._extract_title(page),
                "url": page.get("url"),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time")
            }
            
            # Extract property values based on type
            for prop_name, prop_data in properties.items():
                prop_type = prop_data.get("type")
                
                if prop_type == "title":
                    extracted[prop_name] = self._extract_rich_text(prop_data.get("title", []))
                elif prop_type == "rich_text":
                    extracted[prop_name] = self._extract_rich_text(prop_data.get("rich_text", []))
                elif prop_type == "number":
                    extracted[prop_name] = prop_data.get("number")
                elif prop_type == "select":
                    select_data = prop_data.get("select")
                    extracted[prop_name] = select_data.get("name") if select_data else None
                elif prop_type == "date":
                    date_data = prop_data.get("date")
                    if date_data:
                        extracted[prop_name] = {
                            "start": date_data.get("start"),
                            "end": date_data.get("end")
                        }
                elif prop_type == "checkbox":
                    extracted[prop_name] = prop_data.get("checkbox")
                elif prop_type == "status":
                    status_data = prop_data.get("status")
                    extracted[prop_name] = status_data.get("name") if status_data else None
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting page properties: {e}")
            return {}
    
    def _extract_title(self, page: Dict) -> str:
        """Extract title from a Notion page"""
        try:
            properties = page.get("properties", {})
            for prop_data in properties.values():
                if prop_data.get("type") == "title":
                    return self._extract_rich_text(prop_data.get("title", []))
            return "Untitled"
        except:
            return "Untitled"
    
    def _extract_rich_text(self, rich_text_array: List) -> str:
        """Extract plain text from Notion rich text array"""
        try:
            return "".join([item.get("plain_text", "") for item in rich_text_array])
        except:
            return ""
    
    def _categorize_as_tasks(self, entries: List[Dict]) -> List[Dict]:
        """Categorize entries as tasks with priority and status"""
        tasks = []
        for entry in entries:
            task = {
                "title": entry.get("title", ""),
                "status": entry.get("Status", entry.get("status", "Not Started")),
                "priority": entry.get("Priority", entry.get("priority", "Medium")),
                "due_date": entry.get("Due Date", entry.get("due_date")),
                "project": entry.get("Project", entry.get("project", "")),
                "notes": entry.get("Notes", entry.get("notes", "")),
                "url": entry.get("url")
            }
            tasks.append(task)
        return tasks
    
    def _categorize_as_projects(self, entries: List[Dict]) -> List[Dict]:
        """Categorize entries as projects with timeline and status"""
        projects = []
        for entry in entries:
            project = {
                "name": entry.get("title", ""),
                "status": entry.get("Status", entry.get("status", "Planning")),
                "start_date": entry.get("Start Date", entry.get("start_date")),
                "end_date": entry.get("End Date", entry.get("end_date")),
                "progress": entry.get("Progress", entry.get("progress", 0)),
                "team": entry.get("Team", entry.get("team", [])),
                "description": entry.get("Description", entry.get("description", "")),
                "url": entry.get("url")
            }
            projects.append(project)
        return projects
    
    def create_mood_entry(self, mood: UserMood, database_id: str) -> bool:
        """Create mood entry in Notion database"""
        if not self.client:
            return False
        
        try:
            properties = {
                "Date": {"date": {"start": mood.timestamp.isoformat()}},
                "Mood Score": {"number": mood.mood_score},
                "Stress Level": {"number": mood.stress_level},
                "Notes": {"rich_text": [{"text": {"content": mood.notes}}]}
            }
            
            self.client.pages.create(parent={"database_id": database_id}, properties=properties)
            logger.info("Mood entry created in Notion")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Notion mood entry: {e}")
            return False
    
    def create_weekly_report(self, content: str, page_id: str) -> bool:
        """Create or update weekly wellness report"""
        if not self.client:
            return False
        
        try:
            blocks = [
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "Weekly Wellness Report"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                }
            ]
            
            self.client.blocks.children.append(block_id=page_id, children=blocks)
            logger.info("Weekly report created in Notion")
            return True
            
        except Exception as e:
            logger.error(f"Error creating weekly report: {e}")
            return False

class SlackService:
    """Slack integration service"""
    
    def __init__(self):
        self.client = None
        if SLACK_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Slack client"""
        try:
            token = os.getenv('SLACK_BOT_TOKEN')
            if not token:
                logger.warning("Slack bot token not provided")
                return
            
            self.client = SlackClient(token=token)
            logger.info("Slack client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack: {e}")
    
    def send_wellness_reminder(self, channel: str, message: str) -> bool:
        """Send wellness reminder to Slack channel"""
        if not self.client:
            return False
        
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=message,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": message}
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Take a Break"},
                                "action_id": "take_break"
                            }
                        ]
                    }
                ]
            )
            logger.info("Wellness reminder sent to Slack")
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False

class SpotifyService:
    """Spotify integration service"""
    
    def __init__(self):
        self.client = None
        if SPOTIFY_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Spotify client"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.warning("Spotify credentials not provided")
                return
            
            credentials = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.client = spotipy.Spotify(client_credentials_manager=credentials)
            logger.info("Spotify client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spotify: {e}")
    
    def get_mood_playlist(self, mood_score: int) -> Optional[Dict]:
        """Get playlist recommendation based on mood"""
        if not self.client:
            return None
        
        try:
            # Map mood score to search terms
            if mood_score <= 3:
                query = "sad chill relaxing"
            elif mood_score <= 5:
                query = "calm peaceful meditation"
            elif mood_score <= 7:
                query = "upbeat positive"
            else:
                query = "happy energetic motivation"
            
            results = self.client.search(q=query, type='playlist', limit=1)
            playlists = results['playlists']['items']
            
            if playlists:
                playlist = playlists[0]
                return {
                    'name': playlist['name'],
                    'url': playlist['external_urls']['spotify'],
                    'description': playlist['description']
                }
                
        except Exception as e:
            logger.error(f"Error getting Spotify playlist: {e}")
        
        return None

# Initialize services
calendar_service = GoogleCalendarService()
notion_service = NotionService()
slack_service = SlackService()
spotify_service = SpotifyService()

# Helper functions for LLM-powered tools
def _is_task_overdue(task: Dict) -> bool:
    """Check if a task is overdue"""
    due_date = task.get('due_date')
    if due_date and isinstance(due_date, dict):
        due_start = due_date.get('start')
        if due_start:
            try:
                task_date = datetime.fromisoformat(due_start.replace('Z', '')).date()
                return task_date < datetime.now().date()
            except:
                pass
    return False

def _is_task_due_today(task: Dict, target_date: datetime.date) -> bool:
    """Check if a task is due on the target date"""
    due_date = task.get('due_date')
    if due_date and isinstance(due_date, dict):
        due_start = due_date.get('start')
        if due_start:
            try:
                task_date = datetime.fromisoformat(due_start.replace('Z', '')).date()
                return task_date == target_date
            except:
                pass
    return False

# AI-Powered LangChain Tools

@tool
def notion_calendar_analyzer() -> str:
    """
    Reads Notion databases and analyzes calendar to provide LLM-powered wellness suggestions.
    Uses AI to understand patterns and provide intelligent recommendations.
    
    Returns:
        AI-generated comprehensive analysis with personalized wellness recommendations
    """
    try:
        # Gather raw data
        notion_data = notion_service.get_tasks_and_projects()
        calendar_analysis = calendar_service.analyze_calendar_patterns(days_back=14)
        recent_moods = data_manager.get_recent_moods(7)
        recent_breaks = data_manager.get_recent_breaks(7)
        
        # Prepare data for LLM analysis
        data_summary = {
            "notion_tasks": len(notion_data.get('tasks', [])),
            "notion_projects": len(notion_data.get('projects', [])),
            "overdue_tasks": len([t for t in notion_data.get('tasks', []) if _is_task_overdue(t)]),
            "high_priority_tasks": len([t for t in notion_data.get('tasks', []) if t.get('priority', '').lower() in ['high', 'urgent']]),
            "calendar_events_14d": calendar_analysis.get('total_events', 0),
            "back_to_back_meetings": calendar_analysis.get('stress_indicators', {}).get('back_to_back_meetings', 0),
            "stress_events": len(calendar_analysis.get('stress_indicators', {}).get('high_stress_events', [])),
            "avg_mood_7d": sum(m.mood_score for m in recent_moods) / len(recent_moods) if recent_moods else 0,
            "break_compliance": len([b for b in recent_breaks if b.completed]) / len(recent_breaks) * 100 if recent_breaks else 0
        }
        
        # Use LLM to analyze and provide intelligent insights
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="openai/gpt-oss-20b",
            temperature=0.7
        )
        
        analysis_prompt = f"""
As an AI wellness expert, analyze this person's work-life data and provide intelligent insights:

NOTION TASK DATA:
- Total active tasks: {data_summary['notion_tasks']}
- Active projects: {data_summary['notion_projects']}
- Overdue tasks: {data_summary['overdue_tasks']}
- High priority tasks: {data_summary['high_priority_tasks']}

CALENDAR STRESS ANALYSIS (14 days):
- Total events: {data_summary['calendar_events_14d']}
- Back-to-back meetings: {data_summary['back_to_back_meetings']}
- High-stress events: {data_summary['stress_events']}

WELLNESS TRACKING:
- Average mood (7 days): {data_summary['avg_mood_7d']:.1f}/10
- Break compliance rate: {data_summary['break_compliance']:.1f}%

ADDITIONAL CONTEXT:
- Recent high-stress events: {[event.get('summary') for event in calendar_analysis.get('stress_indicators', {}).get('high_stress_events', [])[:3]]}
- Calendar recommendations: {calendar_analysis.get('recommendations', [])}

Please provide:
1. An intelligent assessment of their current stress/wellness state
2. Key patterns or red flags you notice
3. 3-5 personalized, actionable wellness recommendations
4. Prioritization of which issues to address first

Be empathetic, specific, and focus on practical solutions they can implement today.
"""

        response = llm.invoke(analysis_prompt)
        
        return f"🤖 AI-POWERED WELLNESS ANALYSIS\n{'='*40}\n\n{response.content}"
        
    except Exception as e:
        logger.error(f"Error in notion_calendar_analyzer: {e}")
        return f"Error in AI analysis: {str(e)}"

@tool
def intelligent_break_scheduler(target_date: str = None) -> str:
    """
    Uses AI to intelligently schedule wellness breaks based on workload analysis.
    LLM determines optimal break types, timing, and reasoning.
    
    Args:
        target_date: Date to schedule breaks for (YYYY-MM-DD format, defaults to today)
    
    Returns:
        AI-recommended break schedule with intelligent reasoning
    """
    try:
        # Parse target date
        if target_date:
            try:
                date = datetime.strptime(target_date, "%Y-%m-%d").date()
            except:
                date = datetime.now().date()
        else:
            date = datetime.now().date()
        
        # Gather contextual data
        notion_data = notion_service.get_tasks_and_projects()
        optimal_slots = calendar_service.find_optimal_break_slots(date, duration_minutes=15)
        upcoming_events = calendar_service.get_upcoming_events(hours=24)
        
        # Prepare data for LLM
        context_data = {
            "target_date": date.strftime("%A, %B %d, %Y"),
            "available_slots": [
                {
                    "time": slot['start_time'].strftime("%H:%M"),
                    "duration_available": slot['gap_available'],
                    "context": slot['reason']
                } for slot in optimal_slots[:5]
            ],
            "tasks_today": [
                {
                    "title": task.get('title', ''),
                    "priority": task.get('priority', 'medium'),
                    "status": task.get('status', 'not started')
                } for task in notion_data.get('tasks', [])
                if _is_task_due_today(task, date)
            ][:5],
            "upcoming_meetings": [
                {
                    "title": event.get('summary', ''),
                    "start": event.get('start', {}).get('dateTime', '')
                } for event in upcoming_events[:5]
            ],
            "user_preferences": data_manager.get_best_break_types()
        }
        
        # Use LLM for intelligent scheduling
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        scheduling_prompt = f"""
As an AI wellness coach, design an optimal break schedule for {context_data['target_date']}.

AVAILABLE TIME SLOTS:
{context_data['available_slots']}

TODAY'S TASKS:
{context_data['tasks_today']}

UPCOMING MEETINGS:
{context_data['upcoming_meetings']}

USER'S EFFECTIVE BREAK TYPES (in order of preference):
{context_data['user_preferences']}

Please recommend:
1. Which 2-3 time slots to use for breaks
2. What type of break for each slot (meditation, walk, stretching, breathing, etc.)
3. WHY you chose each break type based on the surrounding activities
4. How to prepare mentally for upcoming challenging tasks/meetings

Format as: 
Time | Break Type | Reasoning | Preparation Tip

Be specific about the psychological and physiological benefits of each break choice.
"""

        ai_response = llm.invoke(scheduling_prompt)
        
        # Schedule actual calendar events for the recommended breaks
        scheduled_count = 0
        for slot in optimal_slots[:3]:  # Limit to 3 breaks
            # Use a default break type (could be enhanced to parse AI response)
            break_type = data_manager.get_best_break_types()[0] if data_manager.get_best_break_types() else "mindfulness"
            
            if calendar_service.insert_break_event(slot['start_time'], 15, break_type):
                scheduled_count += 1
                
                # Log the break
                break_entry = WellnessBreak(
                    timestamp=slot['start_time'],
                    break_type=break_type,
                    duration_minutes=15
                )
                data_manager.add_break(break_entry)
        
        response = f"🧠 AI BREAK SCHEDULING FOR {context_data['target_date']}\n{'='*50}\n\n"
        response += f"✅ Successfully scheduled {scheduled_count} AI-optimized breaks\n\n"
        response += ai_response.content
        
        return response
        
    except Exception as e:
        logger.error(f"Error in intelligent_break_scheduler: {e}")
        return f"Error in AI break scheduling: {str(e)}"

@tool
def ai_stress_predictor_and_advisor(days_ahead: int = 7) -> str:
    """
    Uses AI to predict stress levels and provide proactive wellness strategies.
    LLM analyzes patterns and generates personalized advice.
    
    Args:
        days_ahead: Number of days to analyze ahead (default: 7)
    
    Returns:
        AI-generated stress prediction with proactive wellness strategies
    """
    try:
        # Gather comprehensive data for AI analysis
        notion_data = notion_service.get_tasks_and_projects()
        upcoming_events = calendar_service.get_upcoming_events(hours=days_ahead*24)
        historical_moods = data_manager.get_recent_moods(14)
        historical_breaks = data_manager.get_recent_breaks(14)
        
        # Prepare structured data for LLM
        analysis_data = {
            "upcoming_tasks_by_priority": {
                "high": [t for t in notion_data.get('tasks', []) if t.get('priority', '').lower() in ['high', 'urgent']],
                "medium": [t for t in notion_data.get('tasks', []) if t.get('priority', '').lower() == 'medium'],
                "low": [t for t in notion_data.get('tasks', []) if t.get('priority', '').lower() == 'low']
            },
            "upcoming_events_summary": [
                {
                    "day": datetime.fromisoformat(event.get('start', {}).get('dateTime', '').replace('Z', '+00:00')).strftime("%A"),
                    "title": event.get('summary', ''),
                    "stress_indicators": any(keyword in event.get('summary', '').lower() 
                                           for keyword in ['deadline', 'review', 'presentation', 'interview'])
                } for event in upcoming_events
            ],
            "mood_patterns": {
                "recent_average": sum(m.mood_score for m in historical_moods[-7:]) / len(historical_moods[-7:]) if historical_moods else 5,
                "trend": "improving" if len(historical_moods) >= 2 and historical_moods[-1].mood_score > historical_moods[0].mood_score else "stable",
                "worst_days": [mood.timestamp.strftime("%A") for mood in historical_moods if mood.mood_score <= 4]
            },
            "wellness_habits": {
                "break_consistency": len([b for b in historical_breaks if b.completed]) / len(historical_breaks) * 100 if historical_breaks else 0,
                "most_effective_breaks": data_manager.get_best_break_types()[:3]
            }
        }
        
        # Use LLM for intelligent prediction and advice
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.3-70b-versatile",
            temperature=0.8
        )
        
        prediction_prompt = f"""
As an AI wellness psychologist, analyze this person's data to predict stress levels and provide proactive strategies for the next {days_ahead} days.

TASK WORKLOAD ANALYSIS:
- High priority tasks: {len(analysis_data['upcoming_tasks_by_priority']['high'])}
- Medium priority tasks: {len(analysis_data['upcoming_tasks_by_priority']['medium'])}
- Low priority tasks: {len(analysis_data['upcoming_tasks_by_priority']['low'])}

CALENDAR STRESS INDICATORS:
{analysis_data['upcoming_events_summary']}

MOOD & WELLNESS PATTERNS:
- Recent mood average: {analysis_data['mood_patterns']['recent_average']:.1f}/10
- Mood trend: {analysis_data['mood_patterns']['trend']}
- Historical challenging days: {analysis_data['mood_patterns']['worst_days']}
- Break completion rate: {analysis_data['wellness_habits']['break_consistency']:.1f}%
- Most effective break types: {analysis_data['wellness_habits']['most_effective_breaks']}

Please provide:

1. STRESS PREDICTION (1-10 scale) for each of the next {days_ahead} days with reasoning
2. EARLY WARNING SIGNS to watch for
3. PERSONALIZED PREVENTION STRATEGIES based on their patterns
4. SPECIFIC DAILY RECOMMENDATIONS that account for their workload
5. EMERGENCY WELLNESS PLAN if stress levels spike unexpectedly

Be specific, actionable, and consider their individual patterns and preferences. Focus on prevention rather than just reaction.
"""

        ai_response = llm.invoke(prediction_prompt)
        
        return f"🔮 AI STRESS PREDICTION & WELLNESS STRATEGY\n{'='*50}\n\n{ai_response.content}"
        
    except Exception as e:
        logger.error(f"Error in ai_stress_predictor_and_advisor: {e}")
        return f"Error in AI stress prediction: {str(e)}"

@tool
def ai_notion_wellness_coach(query: str) -> str:
    """
    AI-powered wellness coach that searches Notion and provides intelligent insights.
    Uses LLM to understand content and provide personalized coaching.
    
    Args:
        query: What to search for in Notion (e.g., "analyze my stress patterns", "review my goals")
    
    Returns:
        AI-generated coaching insights based on Notion content analysis
    """
    try:
        # Search Notion for relevant content
        search_results = notion_service.search_notion_content(query, max_results=10)
        
        # Get additional context
        tasks_and_projects = notion_service.get_tasks_and_projects()
        user_profile = data_manager.profile
        
        # Prepare comprehensive context for LLM
        notion_context = {
            "search_query": query,
            "found_pages": len(search_results),
            "relevant_content": [
                {
                    "title": result.get('title', ''),
                    "content_preview": result.get('content', '')[:300] + "..." if result.get('content', '') else "No content",
                    "last_updated": result.get('last_edited_time', '')
                } for result in search_results[:5]
            ],
            "task_overview": {
                "total_tasks": len(tasks_and_projects.get('tasks', [])),
                "total_projects": len(tasks_and_projects.get('projects', [])),
                "overdue_items": len([t for t in tasks_and_projects.get('tasks', []) if _is_task_overdue(t)])
            },
            "wellness_profile": {
                "preferred_activities": user_profile.preferred_break_types,
                "effectiveness_data": user_profile.break_effectiveness,
                "mood_patterns": user_profile.mood_patterns
            }
        }
        
        # Use LLM as intelligent wellness coach
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        coaching_prompt = f"""
You are an AI wellness coach with access to this person's Notion workspace. They asked: "{query}"

NOTION CONTENT FOUND:
{notion_context['relevant_content']}

CURRENT TASK/PROJECT STATUS:
- Total active tasks: {notion_context['task_overview']['total_tasks']}
- Active projects: {notion_context['task_overview']['total_projects']}
- Overdue items: {notion_context['task_overview']['overdue_items']}

WELLNESS PROFILE:
- Preferred wellness activities: {notion_context['wellness_profile']['preferred_activities']}
- Break effectiveness data: {notion_context['wellness_profile']['effectiveness_data']}
- Mood patterns by day: {notion_context['wellness_profile']['mood_patterns']}

As their wellness coach, provide:

1. KEY INSIGHTS from their Notion content related to their query
2. PATTERNS you notice in their work/wellness approach
3. PERSONALIZED RECOMMENDATIONS based on what you found
4. SPECIFIC ACTIONS they can take this week
5. SUGGESTED NOTION IMPROVEMENTS for better wellness tracking

Be supportive, specific, and focus on actionable advice that leverages their existing systems and preferences.
"""

        ai_response = llm.invoke(coaching_prompt)
        
        response = f"🎯 AI WELLNESS COACHING SESSION\n{'='*40}\n"
        response += f"Query: {query}\n"
        response += f"Analyzed {len(search_results)} Notion pages\n\n"
        response += ai_response.content
        
        return response
        
    except Exception as e:
        logger.error(f"Error in ai_notion_wellness_coach: {e}")
        return f"Error in AI wellness coaching: {str(e)}"

@tool 
def ai_daily_wellness_optimizer() -> str:
    """
    AI-powered daily wellness optimization that considers all available data.
    Provides intelligent, context-aware recommendations for the current day.
    
    Returns:
        Comprehensive AI-generated daily wellness plan
    """
    try:
        # Gather comprehensive current context
        now = datetime.now()
        today = now.date()
        
        # Get all relevant data
        today_events = calendar_service.get_events_in_range(
            datetime.combine(today, datetime.min.time()),
            datetime.combine(today, datetime.max.time())
        )
        
        notion_data = notion_service.get_tasks_and_projects()
        recent_mood = data_manager.get_recent_moods(1)
        current_profile = data_manager.profile
        
        # Current day context
        daily_context = {
            "current_time": now.strftime("%H:%M"),
            "day_of_week": today.strftime("%A"),
            "today_events": [
                {
                    "time": event.get('start', {}).get('dateTime', ''),
                    "title": event.get('summary', ''),
                    "duration": "TBD"  # Could calculate if needed
                } for event in today_events
            ],
            "tasks_due_today": [
                task for task in notion_data.get('tasks', [])
                if _is_task_due_today(task, today)
            ],
            "current_mood": recent_mood[0].mood_score if recent_mood else "unknown",
            "historical_patterns": {
                "typical_mood_today": current_profile.mood_patterns.get(today.strftime("%A"), []),
                "best_break_types": current_profile.preferred_break_types,
                "effectiveness_ratings": current_profile.break_effectiveness
            },
            "remaining_day_hours": max(0, 17 - now.hour)  # Assume work ends at 5 PM
        }
        
        # Use LLM for intelligent daily optimization
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        
        optimization_prompt = f"""
You are an AI wellness optimizer. It's currently {daily_context['current_time']} on {daily_context['day_of_week']}. 

TODAY'S SCHEDULE:
{daily_context['today_events']}

TASKS DUE TODAY:
{[task.get('title') for task in daily_context['tasks_due_today']]}

CURRENT STATE:
- Current mood: {daily_context['current_mood']}/10
- Remaining work hours: {daily_context['remaining_day_hours']}
- Historical {daily_context['day_of_week']} mood pattern: {daily_context['historical_patterns']['typical_mood_today']}

WELLNESS PREFERENCES:
- Preferred break types: {daily_context['historical_patterns']['best_break_types']}
- Break effectiveness ratings: {daily_context['historical_patterns']['effectiveness_ratings']}

Please provide:
1. A prioritized wellness plan for the rest of today (breaks, self-care, focus blocks)
2. Specific timing and type of each recommended activity
3. Personalized advice based on mood, workload, and historical patterns
4. Motivation and encouragement for the user

Be concise, actionable, and empathetic. Focus on practical steps the user can take today.
"""

        ai_response = llm.invoke(optimization_prompt)

        response = f"🌞 AI DAILY WELLNESS OPTIMIZER\n{'='*40}\n"
        response += f"Date: {today.strftime('%A, %B %d, %Y')}\n"
        response += f"Current time: {now.strftime('%H:%M')}\n\n"
        response += ai_response.content

        return response

    except Exception as e:
        logger.error(f"Error in ai_daily_wellness_optimizer: {e}")
        return f"Error in AI daily wellness optimization: {str(e)}"

@tool
def descope_signup(email: str, password: str) -> str:
    """
    Signup a new user with email and password using Descope.
    """
    try:
        user = descope_client.auth.password.signup(email, password)
        return f"User signed up successfully: {user}"
    except Exception as e:
        return f"Signup failed: {e}"


@tool
def descope_login(email: str, password: str) -> str:
    """
    Login a user with email and password using Descope.
    """
    try:
        user = descope_client.auth.password.sign_in(email, password)
        return f"User logged in successfully: {user}"
    except Exception as e:
        return f"Login failed: {e}"


@tool
def descope_refresh_session(refresh_token: str) -> str:
    """
    Refresh user session using refresh token.
    """
    try:
        session = descope_client.auth.refresh_session(refresh_token)
        return f"Session refreshed: {session}"
    except Exception as e:
        return f"Refresh failed: {e}"


@tool
def descope_get_user(user_id: str) -> str:
    """
    Fetch user details from Descope by user_id.
    """
    try:
        user = descope_client.management.user.load(user_id)
        return f"User details: {user}"
    except Exception as e:
        return f"Fetch user failed: {e}"


@tool
def descope_update_user(user_id: str, key: str, value: str) -> str:
    """
    Update a user’s details in Descope.
    Example input: user_id, key="name", value="Pranav"
    """
    try:
        descope_client.management.user.update(user_id, {key: value})
        return f"User {user_id} updated: {key} -> {value}"
    except Exception as e:
        return f"Update failed: {e}"


@tool
def descope_delete_user(user_id: str) -> str:
    """
    Delete a user from Descope.
    """
    try:
        descope_client.management.user.delete(user_id)
        return f"User {user_id} deleted successfully."
    except Exception as e:
        return f"Delete failed: {e}"

@tool
def generate_content(prompt: str) -> str:
    """
    Generate an image or video using Replicate API based on the prompt.
    Detects 'image' or 'video' and uses appropriate available model.
    """

    prompt_lower = prompt.lower()

    # === IMAGE GENERATION ===
    if re.search(r"\b(make|generate|create)\b.*\b(image|poster|art|photo|picture)\b", prompt_lower):
        try:
            print("[INFO] Generating Image...")
            output = replicate_client.run(
                "stability-ai/sdxl:db21e45c0568ce013c31508c6303fef9f157b9425fbf3d0d02e000fc3a9f1e61",
                input={"prompt": prompt}
            )
            return f"🖼️ Image generated: {output[0]}"
        except Exception as e:
            return f"❌ Error generating image: {e}"

    # === VIDEO GENERATION ===
    elif re.search(r"\b(make|generate|create)\b.*\b(video|clip|animation|motion)\b", prompt_lower):
        try:
            print("[INFO] Generating Video...")
            output = replicate_client.run(
                "ali-vilab/modelscope-text-to-video-generation",
                input={
                    "prompt": prompt,
                    "seed": 42
                }
            )
            return f"🎞️ Video generated: {output}"
        except Exception as e:
            return f"❌ Error generating video: {e}"

    else:
        return "⚠️ Please specify whether to 'make an image' or 'make a video' in your prompt."

#DB 
PG_SERVICE_NAME = "postgresql-x64-15" 

def start_postgres():
    result = subprocess.run(["net", "start", PG_SERVICE_NAME], capture_output=True, text=True)
    time.sleep(2)
    return result.stdout

#  Stop PostgreSQL service on Windows
def stop_postgres():
    result = subprocess.run(["net", "stop", PG_SERVICE_NAME], capture_output=True, text=True)
    return result.stdout

#  Fetch full schema dynamically from Postgres
def fetch_postgres_schema():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public';
                """)
                tables = cursor.fetchall()
                schema_text = ""
                for (table,) in tables:
                    cursor.execute(f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{table}';
                    """)
                    columns = cursor.fetchall()
                    cols_formatted = ", ".join([f"{c} {t.upper()}" for c, t in columns])
                    schema_text += f"Table: {table}({cols_formatted});\n"
                return schema_text
    except Exception as e:
        return f" Failed to fetch schema: {e}"

#  Convert natural language to SQL using OpenAI
def generate_sql(nl_query: str, schema: str) -> str:
    prompt = f"""
You are a helpful assistant that converts natural language questions into PostgreSQL SQL queries.

IMPORTANT:
- Only return the SQL query.
- Do NOT include any explanation, apologies, or extra text.
- Do NOT prefix with "Here is your SQL:" or anything else.
- Just output valid SQL syntax.

Schema:
{schema}

Natural Language Question:
{nl_query}

SQL Query:
"""

    response = chat.invoke([
        HumanMessage(content=prompt)
    ])
    
    return response.content.strip().split("SQL Query:")[-1].strip()

# Run the generated SQL
def run_sql(sql_query: str):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                if sql_query.strip().lower().startswith("select"):
                    return cur.fetchall()
                else:
                    conn.commit()
                    return f"Executed. Rows affected: {cur.rowcount}"
    except Exception as e:
        return f" SQL Execution Error: {e}"

#  LangChain Tool with Auto Start/Stop Server
@tool("nl_to_postgres", return_direct=True)
def nl_to_postgres(nl_query: str) -> str:
    """Takes a natural language query, starts the PostgreSQL server, runs it, and stops the server."""
    try:
        server_start = start_postgres()
        schema = fetch_postgres_schema()
        if schema.startswith("❌"):
            return schema
        sql = generate_sql(nl_query, schema)
        result = run_sql(sql)
        return f" SQL: {sql}\n📊 Result: {result}"
    finally:
        server_stop = stop_postgres()


PROJECT_ID = "langchain-ad165"
SESSION_ID = "user_session_new"  # This can be username or unique ID
COLLECTION_NAME = "chat_history"

client = firestore.Client(project=PROJECT_ID)

chat_history=FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,

)

with open("config.json", "r") as f:
    config = json.load(f)

def human_sleep(min_time=0.8, max_time=2.5):
    time.sleep(random.uniform(min_time, max_time))
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Tavily search with better configuration
tavily_search_tool = TavilySearch(
    max_results=3,  # Reduced for better performance
    topic='general',
    include_answer=True,
    include_raw_content=False  # Avoid overwhelming the agent
)

# Improved prompt template that's less restrictive
my_prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "chat_history"],
    template="""
You are a helpful AI assistant with access to tools.

Conversation so far:
{chat_history}

Use the following tools to help answer questions:
- get_system_time: Get current date/time
- tavily_search: Search the web for current information
- llm_query: Get general knowledge answers
- research_and_analyze: Combine web search with analysis for comprehensive answers
- apply_linkedin_jobs: Automates applying to LinkedIn jobs using config.json and Playwright.

Think step by step and use tools when needed.

{agent_scratchpad}

Question: {input}
"""
)
def click_next_or_submit(page):
    try:
        next_btn = page.query_selector('button[aria-label*="Next"]')
        if next_btn:
            next_btn.click()
            print(" Clicked Next. ")
            return "next"

        submit_btn = page.query_selector('button[aria-label*="Submit application"]')
        if submit_btn:
            submit_btn.click()
            print(" Application submitted !!")
            return "submit"

        print(" No Next or Submit button.")
        return "done"
    except Exception as e:
        print(f" Error clicking next or submit: {e}")
        return "error"


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=chat_history
)

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from notion_client import Client

@tool
def generate_and_append_note_to_notion(topic: str) -> str:
    """
    Generates AI-written notes on a topic and appends them to a Notion page.
    
    Example:
    Input: Arduino
    Output: Creates notes about Arduino and adds them to Notion.
    """
    try:
        NOTION_TOKEN = os.getenv("NOTION_TOKEN")
        PAGE_ID = os.getenv("NOTION_PAGE_ID")

        if not NOTION_TOKEN or not PAGE_ID:
            return " Error: Missing NOTION_TOKEN or NOTION_PAGE_ID in environment variables."

        # Step 1: Generate notes using LLM
        prompt = f"Write detailed, structured study notes on the topic: {topic}. Use bullet points or sections if helpful."
        notes = llm.invoke(prompt)
        notes_text = notes.content if hasattr(notes, 'content') else str(notes)

        # Step 2: Append to Notion
        notion = Client(auth=NOTION_TOKEN)

        notion.blocks.children.append(
            block_id=PAGE_ID,
            children=[
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": f"Notes on {topic}"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": notes_text}}]
                    }
                }
            ]
        )

        return f" Notes on '{topic}' added to Notion."

    except Exception as e:
        return f" Failed to generate and append notes: {e}"


@tool
def append_to_notion_page(text: str) -> str:
    """
    Appends a paragraph block to a Notion page.
    Input should be plain text. The page must be shared with the integration.
    
    Example:
    Add Summary of today's AI meeting and plans for tomorrow.
    """

    try:
        NOTION_TOKEN = os.getenv("NOTION_TOKEN")
        PAGE_ID = os.getenv("NOTION_PAGE_ID")  # No dashes or with dashes both work

        if not NOTION_TOKEN or not PAGE_ID:
            return " Error: NOTION_TOKEN or NOTION_PAGE_ID is missing in environment variables."

        notion = Client(auth=NOTION_TOKEN)

        notion.blocks.children.append(
            block_id=PAGE_ID,
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": text
                                }
                            }
                        ]
                    }
                }
            ]
        )

        return " Successfully appended text to Notion page."

    except Exception as e:
        return f" Failed to append to Notion page: {e}"

@tool
def create_calendar_event(details: str) -> str:
    """
    Creates a Google Calendar event.
    Expects structured input:
    Title: Meeting with Team
    Date: 2025-06-15
    Time: 14:00
    Duration: 1  # in hours
    Description: Discussion on project milestones
    """

    try:
        # Parse input
        lines = details.strip().split("\n")
        event = {
            "summary": "",
            "description": "",
            "start": {},
            "end": {},
        }

        for line in lines:
            if "Title:" in line:
                event["summary"] = line.split("Title:")[1].strip()
            elif "Date:" in line:
                date = line.split("Date:")[1].strip()
            elif "Time:" in line:
                time_ = line.split("Time:")[1].strip()
            elif "Duration:" in line:
                duration = int(line.split("Duration:")[1].strip())
            elif "Description:" in line:
                event["description"] = line.split("Description:")[1].strip()

        start_datetime = f"{date}T{time_}:00"
        end_hour = int(time_.split(":")[0]) + duration
        end_datetime = f"{date}T{end_hour:02d}:{time_.split(':')[1]}:00"

        event["start"]["dateTime"] = start_datetime
        event["start"]["timeZone"] = "Asia/Kolkata"
        event["end"]["dateTime"] = end_datetime
        event["end"]["timeZone"] = "Asia/Kolkata"

        # Auth
        scopes = ["https://www.googleapis.com/auth/calendar"]
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
        creds = flow.run_local_server(port=3000)
        service = build("calendar", "v3", credentials=creds)

        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f" Event created: {created_event.get('htmlLink')}"

    except Exception as e:
        return f" Failed to create event: {e}"



@tool
def write_email(email_text: str) -> str:
    """
    Sends an email. Expects structured input:
    To: recipient@example.com
    Subject: Your subject here
    Body:
    The body of the email goes here.
    """

    try:
        # Split and parse input
        lines = email_text.strip().split("\n")
        to = next((line.split("To:")[1].strip() for line in lines if line.startswith("To:")), None)
        subject = next((line.split("Subject:")[1].strip() for line in lines if line.startswith("Subject:")), None)
        
        body_index = next((i for i, line in enumerate(lines) if line.strip() == "Body:"), None)
        body = "\n".join(lines[body_index+1:]).strip() if body_index is not None else None

        if not to or not subject or not body:
            return "Error: Email must include 'To:', 'Subject:', and 'Body:'."

        # Credentials
        sender_email = os.getenv("GMAIL_ADDRESS")
        app_password = os.getenv("GMAIL_APP_PASSWORD")

        if not sender_email or not app_password:
            return " Missing Gmail credentials in environment variables."

        # Compose email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, to, msg.as_string())

        return f"Email successfully sent to {to}."

    except Exception as e:
        return f" Failed to send email: {e}"

@tool
def apply_linkedin_jobs(_: str = "") -> str:
    """Applies to LinkedIn jobs using stored config.json and Playwright automation."""
    logs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config.get("headless", False))
        context = browser.new_context()
        page = context.new_page()

        try:
            page.goto("https://www.linkedin.com/login")
            human_sleep()
            page.fill('input[name="session_key"]', config["email"])
            human_sleep()
            page.fill('input[name="session_password"]', config["password"])
            human_sleep()
            page.click('button[type="submit"]')
            human_sleep()
            page.wait_for_selector('a.global-nav__primary-link--active', timeout=0)
            logs.append(" Login successful.")

            page.goto("https://www.linkedin.com/jobs/")
            time.sleep(3)
            search_box = page.get_by_role("combobox", name="Search by title, skill, or")
            search_box.click()
            time.sleep(3)
            search_box.fill(config["search_term"])
            search_box.press("Enter")
            time.sleep(5)
            page.click("//button[@aria-label='Easy Apply filter.']")
            logs.append(" Job search & filter applied.")

            current_page = 1
            job_counter = 0
            max_pages = config.get("max_pages", 5)

            while current_page <= max_pages:
                logs.append(f" Page {current_page}")
                job_listings = page.query_selector_all('//div[contains(@class,"display-flex job-card-container")]')

                if not job_listings:
                    logs.append("No jobs found.")
                    break

                for job in job_listings:
                    try:
                        job_counter += 1
                        logs.append(f" Job {job_counter}")
                        job.click()
                        time.sleep(2)

                        if page.query_selector('span.artdeco-inline-feedback__message:has-text("Applied")'):
                            logs.append(" Already applied. Skipping.")
                            continue

                        easy_apply_button = page.wait_for_selector('button.jobs-apply-button', timeout=5000)
                        easy_apply_button.click()
                        time.sleep(3)

                        # Your helper calls here (inputs, dropdowns, checkboxes, resume)
                        # [Same as existing code: handle_inputs_and_textareas, etc.]

                        while True:
                            result = click_next_or_submit(page)
                            if result in ["submit", "done", "error"]:
                                break
                            human_sleep(2, 3)

                        time.sleep(3)
                    except Exception as job_e:
                        logs.append(f" Error on job {job_counter}: {job_e}")
                        continue

                current_page += 1
                next_page_button = page.query_selector(f'button[aria-label=\"Page {current_page}\"]')
                if next_page_button:
                    next_page_button.click()
                    time.sleep(5)
                else:
                    logs.append(" Finished job pages.")
                    break

        except Exception as e:
            logs.append(f" Script error: {e}")
        finally:
            browser.close()

    return "\n".join(logs)


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get current date and time in specified format"""
    try:
        curr_time = datetime.datetime.now()
        formatted_time = curr_time.strftime(format)
        return f"Current time: {formatted_time}"
    except Exception as e:
        return f"Error getting time: {e}"

@tool
def tavily_search(query: str) -> str:

    """Search the web for current information on the given query"""
    try:
        if not query.strip():
            return "Error: Empty query provided"
        
        result = tavily_search_tool.invoke(query)
        # Format the result better
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result)
        else:
            return f"Search results for '{query}': {result}"
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Search error: {e}"

@tool
def llm_query(query: str) -> str:
    """Get general knowledge answer using the LLM"""
    try:
        if not query.strip():
            return "Error: Empty query provided"
        
        response = llm.invoke(query)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"LLM query error: {e}")
        return f"LLM error: {e}"

@tool
def research_and_analyze(query: str) -> str:
    """Combine web search with LLM analysis for comprehensive answers"""
    try:
        if not query.strip():
            return "Error: Empty query provided"
        
        # Get search results
        search_results = tavily_search_tool.invoke(query)
        
        # Create analysis prompt
        analysis_prompt = f"""
        Based on the following search results about "{query}", provide a comprehensive analysis:
        
        Search Results: {search_results}
        
        Please provide a clear, well-structured answer that synthesizes the information.
        """
        
        # Get LLM analysis
        analysis = llm.invoke(analysis_prompt)
        analysis_text = analysis.content if hasattr(analysis, 'content') else str(analysis)
        
        return f"Research Analysis for '{query}':\n\n{analysis_text}"
        
    except Exception as e:
        logger.error(f"Research and analysis error: {e}")
        return f"Research error: {e}"

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Try to get the standard ReAct prompt, fallback to custom
try:
    prompt_template = hub.pull("hwchase17/react")
    print("Using standard ReAct prompt")
except:
    prompt_template = my_prompt_template
    print("Using custom prompt template")

# Define tools
tools = [get_system_time,ai_notion_wellness_coach,notion_calendar_analyzer,
        intelligent_break_scheduler,
        ai_stress_predictor_and_advisor,
        ai_notion_wellness_coach,
        ai_daily_wellness_optimizer,
        generate_content,
        ai_daily_wellness_optimizer,
        append_to_notion_page,
        descope_login,
        tavily_search,
        llm_query,
        research_and_analyze,
        nl_to_postgres,
        descope_signup,
        apply_linkedin_jobs,
        write_email,
        create_calendar_event,
        generate_and_append_note_to_notion
    ]

# Create agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,  # Prevent infinite loops
    max_execution_time=60,
    memory=memory
    # Timeout after 60 seconds
)

def main():
    print("AI Agent is ready! Type 'exit' to quit.")
    
    print("-" * 100)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or command.")
                continue
            
            print("\nThinking...")
            result = agent_executor.invoke(
                {"input": user_input},
                handle_parsing_errors=True
            )
            
            print(f"\nAgent: {result['output']}")
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(result["output"])
            
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Execution error: {e}")
            print(f"Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    main()