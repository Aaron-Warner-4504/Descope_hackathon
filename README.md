# AI-Powered Mental Health Assistant and Companion

**Team WolfByte**
- Aditya Bajaj
- Rajwardhan Bhandigare  
- Pranav Kodlinge

**Hackathon Theme**: Theme 1: Build a purposeful AI agent

## Project Overview

An intelligent AI companion that serves as a comprehensive mental health assistant, integrating with 10+ APIs to provide personalized wellness support, stress prediction, and automated wellness interventions. Built using LangChain agents with Groq LLM to analyze behavioral patterns, predict stress levels, and proactively suggest personalized wellness strategies.

## What We Built

### Core AI-Powered Features
- **Intelligent Stress Prediction**: Analyzes calendar events, task deadlines, and historical mood data to predict stress levels up to 7 days ahead
- **Smart Break Scheduling**: AI automatically identifies optimal time slots in your calendar and schedules personalized wellness breaks
- **Personalized Wellness Coaching**: Provides context-aware recommendations based on individual patterns, workload, and preferences
- **Mood Pattern Analysis**: Tracks mood trends across days/weeks and correlates with calendar events and task completion
- **Proactive Crisis Prevention**: Detects early warning signs and automatically triggers wellness interventions

### Multi-Platform Integration (10+ APIs)
- **Google Calendar**: Analyzes meeting density, identifies back-to-back meetings, and detects stress-inducing keywords
- **Notion**: Reads task databases, analyzes project deadlines, and tracks completion patterns
- **Slack**: Sends automated wellness reminders and facilitates team mental health check-ins
- **Spotify**: Recommends mood-appropriate playlists based on current emotional state
- **Replicate**: Generates calming images and videos for relaxation sessions
- **Descope**: Secure user authentication and profile management
- **PostgreSQL**: Natural language database queries for wellness data analysis
- **Gmail**: Automated wellness check-ins and professional referral emails
- **LinkedIn**: Job application automation to reduce career-related stress
- **Tavily Search**: Real-time mental health resource discovery

### Intelligent Automation
- **Natural Language Processing**: Conversational interface for all wellness interactions
- **Predictive Analytics**: Machine learning-based stress forecasting
- **Automated Interventions**: Proactive wellness suggestions based on detected patterns
- **Cross-Platform Data Synthesis**: Combines data from multiple sources for holistic analysis

## How to Run It

### Prerequisites
- Python 3.8+
- PostgreSQL database
- API keys for integrated services (see setup guide below)

### Quick Start
```bash
# Clone repository
git clone https://github.com/Aaron-Warner-4504/Descope_hackathon.git


# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright for web automation
playwright install chromium

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Setup database
createdb your_database_name

# Run the assistant
python finaldescope.py
```

### Usage Examples
The assistant responds to natural language commands:

- "Analyze my stress patterns for this week"
- "Schedule optimal wellness breaks for today"
- "Predict my stress levels for next week and suggest prevention strategies"
- "Add meditation notes to my Notion workspace"
- "Send wellness reminders to my team"
- "Create a calming nature video for my break"
- "Apply to software engineering jobs on LinkedIn"

## Tech Stack

### Core AI Infrastructure
- **LangChain**: Multi-agent framework with ReAct pattern
- **Groq LLM**: llama-3.3-70b-versatile for intelligent reasoning
- **PostgreSQL**: Structured data storage with natural language querying
- **Google Cloud Firestore**: Conversation memory and chat history

### API Integrations
- **Google Calendar API**: Schedule analysis and event management
- **Notion API**: Task and project management integration
- **Slack Web API**: Team communication and wellness alerts
- **Spotify Web API**: Music therapy and mood-based recommendations
- **Replicate API**: AI-generated therapeutic content
- **Descope API**: Authentication and user management
- **Gmail API**: Automated email communications
- **Tavily Search API**: Real-time information retrieval
- **LinkedIn (via Playwright)**: Career stress reduction automation

### Development Stack
- **Backend**: Python with asyncio support
- **Web Automation**: Playwright for browser-based interactions
- **Data Processing**: Pandas, NumPy for analytics
- **Authentication**: OAuth 2.0, JWT tokens
- **Environment Management**: python-dotenv
- **Logging**: Structured logging with file persistence

## Setup Guide

### Environment Variables (.env)
```env
# Required
GROQ_API_KEY=your_groq_api_key

# Database
DB_NAME=mental_health_db
DB_USER=your_username
DB_PASSWORD=your_password

# Google Services
GOOGLE_CALENDAR_CREDENTIALS=credentials.json
GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json

# Optional Integrations
NOTION_TOKEN=your_notion_token
NOTION_PAGE_ID=your_page_id
SLACK_BOT_TOKEN=xoxb-your-token
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
DESCOPE_PROJECT_ID=your_project_id
REPLICATE_API_TOKEN=your_replicate_token
GMAIL_ADDRESS=your_email
GMAIL_APP_PASSWORD=your_app_password
TAVILY_API_KEY=your_tavily_key
```

### Required Configuration Files
1. **credentials.json**: Google Calendar OAuth credentials
2. **config.json**: LinkedIn automation settings
3. **serviceAccountKey.json**: Google Cloud Firestore credentials

### Database Schema
The application automatically creates tables for:
- User mood tracking
- Wellness break logs
- Stress prediction history
- Break effectiveness ratings

## Features Documentation

### AI-Powered Analysis Tools
1. **Notion Calendar Analyzer**: Comprehensive stress pattern analysis across tasks and calendar
2. **Intelligent Break Scheduler**: ML-optimized break timing with personalized activity suggestions
3. **Stress Predictor**: 7-day stress forecasting with prevention strategies
4. **Daily Wellness Optimizer**: Real-time wellness plan generation
5. **AI Wellness Coach**: Personalized coaching based on Notion workspace analysis

### Automation Capabilities
- LinkedIn job application automation
- Google Calendar event creation
- Notion page content generation
- Email-based wellness communications
- Slack team wellness coordination

### Data Intelligence
- Mood pattern correlation with calendar events
- Break effectiveness machine learning
- Stress indicator keyword detection
- Workload density analysis
- Team wellness metrics

## Demo Video
[(https://youtu.be/fdfqynm2oRM)]

## What We'd Do With More Time

### Enhanced AI Capabilities
- **Custom ML Models**: Train domain-specific models on mental health datasets for improved accuracy
- **Computer Vision Integration**: Real-time emotion detection via webcam for immediate stress assessment
- **Voice Pattern Analysis**: Speech analysis for early stress detection and mood monitoring
- **Advanced NLP**: Sentiment analysis of communications (emails, Slack) for stress indicators
- **Predictive Modeling**: Long-term mental health trend prediction using time-series analysis

### Advanced Wellness Features
- **Group Therapy Facilitation**: AI-moderated peer support groups
- **Personalized Meditation**: AI-generated guided meditation scripts
- **Cognitive Behavioral Therapy**: Interactive CBT exercises and homework tracking
- **Family Support Systems**: Multi-user family wellness coordination
- **Cultural Adaptation**: Culturally-sensitive wellness approaches for diverse populations

### Enterprise Features
- **Corporate Wellness Dashboard**: Company-wide mental health metrics and insights
- **HR Integration**: Workday, BambooHR integration for employee wellness tracking
- **Compliance Reporting**: HIPAA-compliant data handling and reporting
- **API Platform**: Public API for third-party wellness app integration
- **White-label Solutions**: Customizable deployment for healthcare organizations



## Important Disclaimers

This AI assistant is designed to support wellness and mental health awareness. It is not a replacement for professional mental health care, therapy, or medical treatment. Users experiencing serious mental health concerns should consult qualified healthcare providers.

The system prioritizes user privacy and data security, implementing appropriate safeguards for sensitive health information.

## License

MIT License - See LICENSE file for details

## Contributing

We welcome contributions to improve mental health technology. Please see CONTRIBUTING.md for guidelines.

## Support

For technical support or questions:
- Check troubleshooting section in full documentation
- Review application logs in `mental_health_assistant.log`
- Submit issues via GitHub issue tracker
