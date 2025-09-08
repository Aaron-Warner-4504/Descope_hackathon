# AI-Powered Mental Health Assistant

A comprehensive mental health assistant that integrates with various APIs to provide personalized wellness support, mood tracking, and team wellness features. Uses LangChain with Groq for intelligent decision-making and analysis.

## Features

- 🧠 **AI-Powered Wellness Analysis** - Intelligent insights using LLM
- 📅 **Google Calendar Integration** - Stress pattern analysis and break scheduling
- 📝 **Notion Integration** - Task tracking and wellness journaling
- 💬 **Slack Integration** - Team wellness reminders
- 🎵 **Spotify Integration** - Mood-based playlist recommendations
- 🔐 **Descope Authentication** - User management
- 🎨 **Content Generation** - AI-powered image/video creation
- 📧 **Email Automation** - Automated email sending
- 🔍 **Web Search** - Real-time information retrieval
- 💾 **PostgreSQL Integration** - Natural language to SQL queries
- 🌐 **Web Automation** - LinkedIn job application automation

## Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- PostgreSQL installed and running
- A Google account (for Calendar integration)
- Various API keys (see Environment Variables section)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd mental-health-assistant
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright (for LinkedIn automation)

```bash
playwright install chromium
```

## Environment Variables Setup

Create a `.env` file in the project root with the following variables:

```env
# Required - Core AI
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DB_NAME=your_database_name
DB_USER=your_db_username
DB_PASSWORD=your_db_password

# Google Services (Optional)
GOOGLE_CALENDAR_CREDENTIALS=path/to/credentials.json

# Notion (Optional)
NOTION_TOKEN=your_notion_integration_token
NOTION_PAGE_ID=your_notion_page_id
NOTION_MOOD_DATABASE_ID=your_mood_database_id
NOTION_WEEKLY_REPORTS_PAGE_ID=your_reports_page_id

# Slack (Optional)
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

# Spotify (Optional)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Descope Authentication (Optional)
DESCOPE_PROJECT_ID=your_descope_project_id
DESCOPE_MANAGEMENT_KEY=your_descope_management_key

# Content Generation (Optional)
REPLICATE_API_TOKEN=your_replicate_api_token

# Email (Optional)
GMAIL_ADDRESS=your_gmail_address
GMAIL_APP_PASSWORD=your_gmail_app_password

# Search (Optional)
TAVILY_API_KEY=your_tavily_api_key

# Google Cloud Firestore
GOOGLE_APPLICATION_CREDENTIALS=path/to/serviceAccountKey.json
```

## API Keys and Service Setup

### 1. Groq API (Required)

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up/login and create an API key
3. Add to `.env` as `GROQ_API_KEY`

### 2. Google Calendar (Optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Calendar API
4. Create credentials (OAuth 2.0) and download JSON
5. Save as `credentials.json` in project root
6. Set path in `.env` as `GOOGLE_CALENDAR_CREDENTIALS`

### 3. Notion (Optional)

1. Go to [Notion Developers](https://developers.notion.com/)
2. Create a new integration
3. Copy the integration token
4. Share your Notion pages/databases with the integration
5. Get page/database IDs from URLs
6. Add to `.env`

### 4. Slack (Optional)

1. Go to [Slack API](https://api.slack.com/apps)
2. Create a new app
3. Add bot token scopes: `chat:write`, `channels:read`
4. Install app to workspace
5. Copy bot token (starts with `xoxb-`)

### 5. Spotify (Optional)

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app
3. Get Client ID and Client Secret
4. Add to `.env`

### 6. Other Services

- **Descope**: Sign up at [Descope](https://www.descope.com/)
- **Replicate**: Get API key from [Replicate](https://replicate.com/)
- **Tavily**: Get API key from [Tavily](https://tavily.com/)

## Database Setup

### 1. Install PostgreSQL

Download and install from [PostgreSQL Official Site](https://www.postgresql.org/download/)

### 2. Create Database

```sql
-- Connect to PostgreSQL as superuser
psql -U postgres

-- Create database
CREATE DATABASE your_database_name;

-- Create user
CREATE USER your_db_username WITH PASSWORD 'your_db_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_db_username;
```

### 3. Update Database Configuration

Update the `.env` file with your database credentials.

## Configuration Files

### 1. Create `config.json` (for LinkedIn automation)

```json
{
  "email": "your_linkedin_email@example.com",
  "password": "your_linkedin_password",
  "search_term": "python developer",
  "headless": false,
  "max_pages": 5
}
```

### 2. Google Cloud Setup (for Firestore)

1. Create a Google Cloud project
2. Enable Firestore API
3. Create a service account
4. Download service account key JSON
5. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

## What We Built

An intelligent AI companion that serves as a comprehensive mental health assistant with the following capabilities:

### Core AI-Powered Features
- **Intelligent Stress Prediction**: Uses calendar analysis and task patterns to predict stress levels 7 days ahead
- **Personalized Wellness Coaching**: AI-driven recommendations based on individual patterns and preferences
- **Smart Break Scheduling**: Automatically finds optimal times for wellness breaks and schedules them
- **Mood Pattern Analysis**: Tracks and analyzes mood patterns to provide actionable insights
- **Proactive Interventions**: Sends automated wellness reminders and suggestions via Slack/email

### Multi-Platform Integration
- **Calendar Intelligence**: Google Calendar analysis for stress pattern detection
- **Task Management**: Notion integration for project and task stress analysis
- **Team Wellness**: Slack integration for team mental health support
- **Content Therapy**: AI-generated relaxation content (images/videos) via Replicate
- **Music Therapy**: Spotify playlist recommendations based on mood
- **Email Automation**: Automated wellness check-ins and reminders

## How to Run It

### Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Configure PostgreSQL database
5. Run: `python finaldescope.py`

### Usage Examples
The assistant responds to natural language commands:

- **Wellness Analysis**: "Analyze my stress patterns and workload"
- **Break Scheduling**: "Schedule optimal wellness breaks for today"
- **Notion Integration**: "Add mental health notes to my Notion workspace"
- **Stress Prediction**: "Predict my stress levels for next week"
- **Calendar Management**: "Create a meditation reminder for tomorrow"
- **Team Support**: "Send wellness reminders to my Slack team"
- **Content Generation**: "Create a calming nature image for relaxation"

## Tech Stack

### Required Technologies
- **Core AI**: LangChain + Groq LLM (llama-3.3-70b-versatile)
- **Database**: PostgreSQL with natural language querying
- **Conversation Memory**: Google Cloud Firestore
- **Web Automation**: Playwright for LinkedIn integration

### API Integrations (10+ Services)
- **Google Calendar API**: Schedule analysis and break scheduling
- **Notion API**: Task and project management integration
- **Slack API**: Team wellness communications
- **Spotify API**: Mood-based music recommendations
- **Replicate API**: AI content generation (images/videos)
- **Tavily Search API**: Real-time information retrieval
- **Descope Auth**: User authentication and management
- **Gmail API**: Automated email communications

### Development Stack
- **Backend**: Python 3.8+
- **AI Framework**: LangChain with ReAct agents
- **Data Processing**: Pandas, NumPy
- **Authentication**: OAuth 2.0, JWT tokens
- **Automation**: Playwright, Selenium

## Demo Video
[Demo Video Link - To be added]

## What We'd Do With More Time

### Enhanced AI Capabilities
- **Deep Learning Models**: Train custom models on mental health data for better predictions
- **Emotion Recognition**: Add computer vision for real-time emotion detection via webcam
- **Voice Analysis**: Implement speech pattern analysis for stress detection
- **Predictive Analytics**: More sophisticated ML models for longer-term mental health trends

### Extended Integrations
- **Wearable Devices**: Integration with Fitbit, Apple Watch for biometric stress indicators
- **Microsoft Teams**: Expand team wellness features to more platforms
- **Zoom/Meet**: Calendar integration with video call stress analysis
- **Health Apps**: Integration with Apple Health, Google Fit for holistic wellness

### Advanced Features
- **Crisis Detection**: AI-powered detection of mental health emergencies with automatic professional referrals
- **Group Therapy**: AI-facilitated group wellness sessions and peer support
- **Therapy Chatbot**: More sophisticated conversational AI trained on therapeutic techniques
- **Personalized Content**: AI-generated meditation scripts, breathing exercises tailored to individual needs

### Professional Integration
- **Therapist Dashboard**: Professional interface for mental health practitioners
- **Clinical Reports**: Automated generation of progress reports for healthcare providers
- **Insurance Integration**: Connect with health insurance for wellness program benefits
- **Corporate Wellness**: Enterprise-grade features for company-wide mental health programs

### Mobile & Accessibility
- **Mobile App**: Native iOS/Android app with offline capabilities
- **Voice Interface**: Hands-free interaction via voice commands
- **Accessibility Features**: Support for users with disabilities
- **Multi-language**: Support for multiple languages and cultural contexts

## Features Documentation

### AI-Powered Tools

1. **Notion Calendar Analyzer**: Analyzes your tasks and calendar for stress patterns
2. **Intelligent Break Scheduler**: AI-optimized break scheduling
3. **Stress Predictor**: Predicts stress levels and provides strategies
4. **Daily Wellness Optimizer**: Comprehensive daily wellness planning

### Automation Features

1. **LinkedIn Job Applications**: Automated job applications with Playwright
2. **Email Automation**: Send emails via Gmail
3. **Calendar Management**: Create and manage Google Calendar events
4. **Notion Integration**: Read/write to Notion databases and pages

### Wellness Tracking

- Mood tracking with persistence
- Break effectiveness analysis
- Stress pattern recognition
- Personalized recommendations

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed with `pip install -r requirements.txt`

2. **Database Connection Error**: 
   - Ensure PostgreSQL is running
   - Check database credentials in `.env`
   - Verify database exists

3. **Google Calendar Auth Error**:
   - Check credentials.json path
   - Ensure Calendar API is enabled
   - Run authentication flow

4. **Notion API Error**:
   - Verify integration token
   - Check if pages are shared with integration
   - Validate page/database IDs

5. **LinkedIn Automation Issues**:
   - Update `config.json` with correct credentials
   - Check if Playwright browser is installed
   - Verify LinkedIn account access

### Performance Tips

1. **Memory Usage**: The assistant stores conversation history in Firestore
2. **API Limits**: Be aware of rate limits for various APIs
3. **Local Storage**: Wellness data is stored locally in `mental_health_data/`

## Security Considerations

1. **Environment Variables**: Never commit `.env` file to version control
2. **Credentials**: Store all API keys securely
3. **Database**: Use strong passwords for database access
4. **LinkedIn**: Use application-specific passwords when possible

## Project Structure

```
mental-health-assistant/
├── finaldescope.py          # Main application file
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (create this)
├── config.json            # LinkedIn automation config (create this)
├── credentials.json       # Google Calendar credentials (create this)
├── mental_health_data/    # Local wellness data storage
├── mental_health_assistant.log  # Application logs
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `mental_health_assistant.log`
3. Create an issue in the repository

## Version History

- **v2.0**: AI-Powered Edition with LangChain integration
- **v1.0**: Initial release with basic wellness features

---

**Note**: This assistant is for wellness support and should not replace professional mental health care. If you're experiencing serious mental health issues, please consult with a qualified healthcare provider.
