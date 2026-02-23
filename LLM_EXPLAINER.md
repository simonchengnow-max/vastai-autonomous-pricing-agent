# LLM Explainer - Generative AI Layer for Pricing Decisions

## Overview

The LLM Explainer adds a Generative AI layer to the Vast.ai pricing agent, translating complex ML pricing decisions into plain English explanations that executives and business users can understand.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Explainer Layer                          │
│  Transforms ML predictions → Human-readable narratives          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Chutes.ai LLM API (DeepSeek-V3)                    │
│  • Primary: deepseek-ai/DeepSeek-V3-0324                        │
│  • Fallback: unsloth/Llama-3.3-70B-Instruct                     │
│  • Auth: X-API-Key or Bearer token                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Rule-Based Fallback                           │
│  Graceful degradation when API unavailable/rate-limited         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Individual Recommendation Explanations**
Converts technical ML output into business-focused narratives:

**Input (ML Model):**
```json
{
  "gpu_type": "RTX 5090",
  "current_market_median": 1.12,
  "ml_predicted_price": 0.9182,
  "action": "decrease",
  "reason": "Current price 22.0% above ML prediction",
  "confidence": 0.75
}
```

**Output (LLM Explanation):**
```
The RTX 5090 is currently priced 22.0% above our ML model's prediction 
of $0.92/hr. Reducing from the market median of $1.12/hr will improve 
competitiveness and increase rental probability in a market with 5 
competing offers.
```

### 2. **Market Narrative Generation**
Creates executive summaries of overall market conditions:

```
Vast.ai market analysis covers 64 active GPU offers across 22 GPU types. 
Our ML pricing model recommends 10 price decreases to improve competitiveness, 
10 increases to capture demand, and 2 holds for optimal positioning. 
Average market price is $1.51/hr with opportunities in H100 NVL, RTX 5060 Ti, 
RTX 4070S Ti segments.
```

### 3. **Graceful Fallback System**
- **Primary**: Chutes.ai LLM (when available)
- **Fallback**: Rule-based explanations (when API fails/rate-limited)
- **Result**: 100% uptime for explanations

## Usage

### Basic Usage

```python
from llm_explainer import PricingExplainer

# Initialize
explainer = PricingExplainer()

# Generate explanations for top 5 recommendations
explanations = explainer.explain_all_recommendations(top_n=5)

# Generate market summary
market_summary = explainer.generate_market_summary()

# Clean up
explainer.close()
```

### Standalone Demo

```bash
python code/vastai-pricing/llm_explainer.py
```

Output:
- Console: Formatted explanations
- File: `code/vastai-pricing/data/llm_explanations.json`

### Integration with Pricing Agent

```python
from pricing_agent import VastAIPricingAgent
from llm_explainer import PricingExplainer

# Run pricing agent
agent = VastAIPricingAgent(api_key="your_key")
recommendations = agent.get_recommendations()

# Generate explanations
explainer = PricingExplainer()
explanations = explainer.explain_all_recommendations()

# Both now available for dashboard/reporting
```

## API Configuration

### Chutes.ai API
- **Base URL**: `https://llm.chutes.ai/v1`
- **Endpoint**: `/chat/completions` (OpenAI-compatible)
- **Models**: 
  - Primary: `deepseek-ai/DeepSeek-V3-0324`
  - Fallback: `unsloth/Llama-3.3-70B-Instruct`
- **Authentication**: Bearer token or X-API-Key header
- **Cost**: Free tier available

### Custom LLM Configuration

```python
explainer = PricingExplainer(
    api_key="your_api_key",
    base_url="https://custom-llm-api.com/v1",
    model="custom/model-name",
    fallback_model="custom/fallback-model"
)
```

## Output Format

### llm_explanations.json

```json
{
  "timestamp": "2026-02-23T03:56:30.699044",
  "market_summary": "Executive summary of market conditions...",
  "explained_recommendations": [
    {
      "gpu_type": "RTX PRO 6000 S",
      "action": "decrease",
      "current_price": 2.4667,
      "recommended_price": 0.9931,
      "confidence": 0.75,
      "llm_explanation": "The RTX PRO 6000 S is currently priced..."
    }
  ]
}
```

## Prompt Engineering

### Recommendation Explanation Prompt

```
Explain this GPU pricing recommendation in 2-3 sentences:

GPU: {gpu_type}
Current Market Median: ${current_median:.2f}/hr
ML Model Recommendation: ${ml_price:.2f}/hr
Action: {action.upper()}
Reason: {reason}
Confidence: {confidence:.0%}
Market Sample Size: {sample_size} offers
Estimated Impact: {impact}

Write a concise explanation that a business executive would understand, 
focusing on WHY this price change makes strategic sense based on market conditions.
```

### Market Summary Prompt

```
Write a 3-4 sentence executive summary of the current GPU rental market:

Total Market Offers: {total_offers}
GPU Types Analyzed: {total_recs}
Sample GPUs: {gpu_types}
Average Market Price: ${avg_market_price:.2f}/hr
Recommended Actions: {decrease} decrease, {increase} increase, {hold} hold

Write this for a business dashboard header - focus on market trends, 
pricing opportunities, and strategic implications.
```

## Error Handling

### API Failures
```python
# Automatic retry with fallback model
explanation = self._call_llm(prompt)  # Tries DeepSeek-V3
# If fails, automatically tries Llama-3.3-70B
# If both fail, returns None → triggers rule-based fallback
```

### Rate Limiting (429 errors)
- Gracefully falls back to rule-based explanations
- No user-facing errors
- System continues to function

### Authentication Errors (401 errors)
- Tries both Bearer and X-API-Key auth methods
- Falls back to rule-based on failure
- Logs error for debugging

## Performance

### Typical API Response Times
- Market Summary: ~2-3 seconds
- Single Explanation: ~1-2 seconds
- Batch of 5: ~6-10 seconds (sequential)

### Rate Limits
- Free tier: Variable based on usage
- Recommendation: Generate explanations in batches
- Current implementation: Top 5 recommendations only

## Future Enhancements

### Planned Features
1. **Batch API Calls**: Parallel processing for faster generation
2. **Caching**: Store explanations to reduce API calls
3. **Confidence Scoring**: LLM confidence in explanation quality
4. **Multi-language**: Generate explanations in multiple languages
5. **Custom Tone**: Adjust formality/technical depth per audience

### Dashboard Integration (Next Phase)
The LLM explanations are designed to feed directly into:
- Real-time pricing dashboard
- Email/Slack notifications
- Executive reports
- API endpoints for web UI

## Examples

### Example 1: Price Decrease Recommendation

**ML Output:**
- GPU: H200
- Current: $4.24/hr
- Recommended: $1.71/hr
- Action: decrease (-59.7%)

**LLM Explanation:**
```
The H200 is currently priced 148.5% above our ML model's prediction of $1.71/hr. 
Reducing from the market median of $4.24/hr will improve competitiveness and 
increase rental probability in a market with 6 competing offers.
```

### Example 2: Price Increase Recommendation

**ML Output:**
- GPU: RTX PRO 6000 WS
- Current: $0.85/hr
- Recommended: $1.42/hr
- Action: increase (+66.9%)

**LLM Explanation:**
```
The RTX PRO 6000 WS shows strong demand with our ML model recommending $1.42/hr, 
which is 39.8% above the current market median of $0.85/hr. This price increase 
captures additional revenue while maintaining competitive positioning.
```

## Files

```
code/vastai-pricing/
├── llm_explainer.py              # Main implementation
├── LLM_EXPLAINER.md              # This documentation
├── data/
│   ├── agent_recommendations.json # Input: ML recommendations
│   ├── market_snapshot.json       # Input: Market data
│   └── llm_explanations.json      # Output: LLM explanations
```

## Dependencies

```python
import json
import httpx  # HTTP client for API calls
from datetime import datetime
from typing import Dict, List, Optional
```

## Testing

```bash
# Run demo
python code/vastai-pricing/llm_explainer.py

# Check output
cat code/vastai-pricing/data/llm_explanations.json

# Verify explanations are generated (either LLM or fallback)
grep "llm_explanation" code/vastai-pricing/data/llm_explanations.json
```

## Conclusion

The LLM Explainer successfully bridges the gap between machine learning predictions and human decision-making by providing clear, actionable explanations of pricing recommendations. With robust fallback mechanisms and efficient API usage, it's production-ready and designed for seamless dashboard integration.
