# OpenAI Integration Setup Guide

## 🤖 AI-Powered Flight Analytics

The Flight Schedule Optimization dashboard now includes advanced AI capabilities powered by OpenAI's GPT models.

### Features

- **🧠 Intelligent Data Analysis**: AI-powered insights from flight data
- **🎯 Optimization Strategies**: AI-generated recommendations for operational improvements  
- **🔮 Predictive Analytics**: Delay forecasting and trend analysis
- **💬 Smart Query Processing**: Natural language interface for data queries
- **🚨 Operational Alerts**: Intelligent alert generation for critical issues

### Setup Instructions

1. **Get OpenAI API Key**
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create an account or sign in
   - Generate a new API key

2. **Configure Environment**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit .env and add your API key
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Install Dependencies**
   ```bash
   pip install openai>=1.0.0 python-dotenv
   ```

4. **Run Dashboard**
   ```bash
   streamlit run app/main.py
   ```

### Usage

1. Navigate to the **🤖 AI Insights** tab
2. Select your preferred analysis type:
   - **Data Insights**: Comprehensive analysis of flight patterns
   - **Optimization Strategy**: Strategic recommendations for improvements
   - **Delay Predictions**: Future delay forecasting
   - **Smart Query**: Ask questions in natural language
   - **Operational Alerts**: Critical issue identification

3. Click **🚀 Generate AI Analysis** to get AI-powered insights

### Example Queries

- "What are the main causes of delays in our data?"
- "Which routes have the highest delay risk?"
- "How can we improve on-time performance?"
- "What patterns do you see in our flight operations?"

### Cost Considerations

- Uses GPT-3.5-turbo model (cost-effective)
- Typical cost: $0.002 per 1K tokens
- Average analysis: ~$0.01-$0.05 per request
- Set usage limits in your OpenAI account for cost control

### Troubleshooting

**"OpenAI not available" error:**
- Check API key is correctly set in .env file
- Verify internet connection
- Confirm OpenAI account has available credits

**Rate limiting:**
- OpenAI has rate limits for API calls
- Free tier: 3 requests per minute
- Paid tier: Higher limits available

### Security

- Never commit your API key to version control
- Use environment variables only
- Rotate API keys regularly
- Monitor usage in OpenAI dashboard
