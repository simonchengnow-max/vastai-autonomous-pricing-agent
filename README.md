# Vast.ai Autonomous Pricing Agent

A complete autonomous pricing system for Vast.ai GPU hosts that monitors the market in real-time and recommends optimal price adjustments using machine learning and rule-based decision logic.

## Overview

The pricing agent combines:
- **Market Monitoring** - Live data fetching from Vast.ai API
- **ML Price Prediction** - Random Forest model trained on market data
- **Trend Analysis** - Detects demand spikes and price drops
- **Decision Engine** - Hybrid ML + rules-based recommendations
- **LLM Explainer** - Generative AI translates ML decisions to plain English
- **Automated Logging** - Complete audit trail of all decisions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS PRICING AGENT                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Market     │─────>│   Trend      │─────>│  Decision    │  │
│  │   Monitor    │      │   Analyzer   │      │  Engine      │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                      │          │
│         v                     v                      v          │
│  Fetch Live Data      Compare vs History     ML + Rules        │
│  Every 15min          Detect Spikes/Drops    Generate Actions  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ML PRICING MODEL (Random Forest)            │  │
│  │  - Trained on 63 GPU configurations                      │  │
│  │  - 95% R² accuracy on test set                          │  │
│  │  - Confidence intervals via tree ensemble               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  OUTPUT: Recommendations JSON + JSONL Log                       │
└─────────────────────────────────────────────────────────────────┘
```

## Files Structure

```
code/vastai-pricing/
├── pricing_agent.py              # Main autonomous agent (PricingAgent class)
├── ml_pricing_model.py           # ML model training script
├── llm_explainer.py              # LLM-powered explanation generator
├── models/
│   ├── pricing_model.pkl         # Trained Random Forest model
│   └── model_results.json        # Model performance metrics
├── data/
│   ├── market_snapshot.json      # Latest market data
│   ├── market_analysis.json      # Market analysis results
│   ├── agent_recommendations.json # Latest pricing recommendations
│   └── llm_explanations.json     # Plain English explanations
├── logs/
│   └── agent_run_log.jsonl       # Complete audit trail
├── README.md                     # This file
└── LLM_EXPLAINER.md              # LLM explainer documentation
```

## Features

### 1. Market Monitor
- Fetches live GPU offers from Vast.ai API every N minutes (default: 15)
- Tracks verified, rentable GPU hosts
- Maintains rolling history of last 10 snapshots
- Query: `{"verified": {"eq": true}, "rentable": {"eq": true}}`

### 2. Trend Analyzer
- Compares current vs previous market snapshots
- Detects **demand spikes** (>20% increase in rentals)
- Detects **price drops** (>10% decrease in median price)
- Analyzes statistics by GPU type (median, mean, min, max, std)

### 3. ML Price Optimizer
- Uses trained Random Forest model (95% R² accuracy)
- Predicts optimal price for each GPU configuration
- Provides confidence intervals (95% CI using tree ensemble std)
- Features: gpu_ram, cpu_ram, num_gpus, reliability, bandwidth, dlperf, etc.

### 4. Decision Engine (Hybrid ML + Rules)

Combines ML predictions with market trend rules to generate pricing recommendations.

### 5. LLM Explainer (Generative AI Layer)
- Translates technical ML predictions into plain English business narratives
- Uses Chutes.ai API (DeepSeek-V3 or Llama-3.3-70B)
- Generates individual recommendation explanations (2-3 sentences)
- Creates executive market summaries for dashboards
- Graceful fallback to rule-based explanations if API unavailable
- See [LLM_EXPLAINER.md](LLM_EXPLAINER.md) for details

### 6. Decision Rules

**Rule 1: Market Price Drop**
- If market-wide price drop >10% detected
- AND current price < ML prediction * 0.9
- → ACTION: Decrease price by 5%
- Confidence: +0.2

**Rule 2: Demand Spike**
- If rentals increase >20%
- → ACTION: Increase price by 10%
- Confidence: +0.3

**Rule 3: ML Prediction Alignment**
- If current price >15% above ML prediction
- → ACTION: Decrease to ML predicted price
- Confidence: +0.25
- If current price <10% below ML prediction
- → ACTION: Increase to ML predicted price (leaving money on table)
- Confidence: +0.25

**Rule 4: Market Position**
- If current price >15% above market average
- → ACTION: Decrease to market average
- Confidence: +0.15

**Guardrails:**
- Minimum: $0.10/hr
- Maximum: $50.00/hr
- All recommendations constrained within bounds

### 5. Output & Logging

**agent_recommendations.json:**
```json
{
  "timestamp": "2026-02-23T03:52:53",
  "total_recommendations": 22,
  "recommendations": [
    {
      "gpu_type": "RTX 5090",
      "current_market_median": 1.1200,
      "current_market_mean": 1.1120,
      "ml_predicted_price": 0.9182,
      "ml_confidence_interval": {"lower": -0.2716, "upper": 2.1081},
      "recommended_price": 0.9182,
      "action": "decrease",
      "reason": "Current price 22.0% above ML prediction",
      "confidence": 0.75,
      "estimated_impact": "-$0.202/hr, increase competitiveness",
      "market_sample_size": 5,
      "timestamp": "2026-02-23T03:52:52"
    }
  ]
}
```

**agent_run_log.jsonl:**
```json
{"timestamp": "2026-02-23T03:52:53", "market_total_offers": 64, "trends": {"demand_spike_detected": false, "price_drop_detected": false, "gpu_types_analyzed": 22}, "recommendations": {"total": 22, "increase": 10, "decrease": 10, "hold": 2, "high_confidence": 20}}
```

## Usage

### Quick Start: Single Cycle

```python
from pricing_agent import PricingAgent

# Initialize agent
agent = PricingAgent(
    api_key="your_vast_api_key",
    monitor_interval=15  # minutes
)

# Run one cycle
recommendations = agent.run_cycle()

# View results
agent.print_summary(recommendations)
```

### With LLM Explanations

```python
from pricing_agent import PricingAgent
from llm_explainer import PricingExplainer

# Run pricing agent
agent = PricingAgent(api_key="your_vast_api_key")
recommendations = agent.run_cycle()

# Generate plain English explanations
explainer = PricingExplainer()
explanations = explainer.explain_all_recommendations(top_n=5)
market_summary = explainer.generate_market_summary()

print(f"\nMarket Summary:\n{market_summary}\n")
for exp in explanations:
    print(f"{exp['gpu_type']}: {exp['llm_explanation']}")

explainer.close()
```

### Continuous Monitoring

```python
# Run continuously every 15 minutes
agent.run_continuous(max_cycles=None)  # Runs forever

# Or run limited cycles
agent.run_continuous(max_cycles=10)  # Run 10 cycles then stop
```

### Command Line Demo

```bash
cd /home/user/files/code/vastai-pricing
python pricing_agent.py
```

## Demo Results

Latest run (2026-02-23 03:52:53):
- **Total Offers Analyzed:** 64
- **GPU Types:** 22
- **Recommendations Generated:** 22
  - INCREASE: 10 GPUs (underpriced, leaving money on table)
  - DECREASE: 10 GPUs (overpriced, losing competitiveness)
  - HOLD: 2 GPUs (optimal pricing)
- **High Confidence (>70%):** 20 recommendations

### Top Recommendations

1. **RTX PRO 6000 S** - DECREASE from $2.47/hr to $0.99/hr (75% confidence)
   - Current price 148% above ML prediction
   - Impact: Increase competitiveness significantly

2. **H200** - DECREASE from $4.24/hr to $1.71/hr (75% confidence)
   - Current price 148% above ML prediction
   - Impact: Major price adjustment needed

3. **RTX 5090** - DECREASE from $1.12/hr to $0.92/hr (75% confidence)
   - Current price 22% above ML prediction
   - Impact: Minor adjustment for better positioning

4. **RTX PRO 6000 WS** - INCREASE from $0.85/hr to $1.42/hr (75% confidence)
   - Current price 40% below ML prediction
   - Impact: +$0.56/hr revenue opportunity

## Model Performance

**Best Model:** Random Forest
- **Test R²:** 0.953 (95.3% variance explained)
- **Test MAE:** $0.27/hr
- **Test RMSE:** $0.48/hr
- **Training Samples:** 50
- **Test Samples:** 13

**Top Features (Importance):**
1. `dlperf` (80.1%) - Deep learning performance
2. `cpu_ram` (14.8%) - System memory
3. `disk_space` (1.1%) - Storage capacity
4. `num_gpus` (1.0%) - GPU count

## Configuration

Edit constants in `pricing_agent.py`:

```python
DEFAULT_MONITOR_INTERVAL = 15  # minutes between checks
MIN_PRICE = 0.10              # $/hour minimum
MAX_PRICE = 50.00             # $/hour maximum
PRICE_DROP_THRESHOLD = 0.10   # 10% drop triggers action
PRICE_SPIKE_THRESHOLD = 0.20  # 20% demand spike triggers action
OVERPRICED_THRESHOLD = 0.15   # 15% above median = overpriced
UNDERPRICED_THRESHOLD = 0.10  # 10% below median = underpriced
```

## API Integration

**Vast.ai API Endpoint:**
```
GET https://console.vast.ai/api/v0/bundles/
Query: {"verified": {"eq": true}, "rentable": {"eq": true}}
```

**Required API Key:**
- Set in `VAST_API_KEY` constant or pass to `PricingAgent()` constructor
- Current key: `c6a50f0314c9b32c6a27ee3920fe71115bf468100aa4825fdb29afb06f9a3c4a`

## Next Steps (Roadmap)

1. **Generative AI Layer** - LLM explains pricing decisions in natural language
2. **Dashboard** - Real-time web UI showing live pricing decisions + reasoning
3. **Governance Layer** - Advanced guardrails, audit trail, decision logging
4. **Automated Execution** - Auto-apply price changes via Vast.ai API
5. **A/B Testing** - Test different pricing strategies
6. **Revenue Tracking** - Monitor actual revenue impact of changes

## Dependencies

```bash
pip install pandas numpy scikit-learn requests
```

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- requests >= 2.28.0

## License

MIT License - Internal tool for Vast.ai GPU host optimization

## Author

Built by Code Agent (Nebula AI)
Date: 2026-02-23
