#!/usr/bin/env python3
"""
Vast.ai Autonomous Pricing Agent
Monitors the market and recommends price adjustments automatically
"""

import json
import pickle
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/user/files/code/vastai-pricing")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
VAST_API_BASE = "https://console.vast.ai/api/v0"
VAST_API_KEY = "c6a50f0314c9b32c6a27ee3920fe71115bf468100aa4825fdb29afb06f9a3c4a"

# Agent Configuration
DEFAULT_MONITOR_INTERVAL = 15  # minutes
MIN_PRICE = 0.10  # $/hour
MAX_PRICE = 50.00  # $/hour
PRICE_DROP_THRESHOLD = 0.10  # 10% drop triggers recommendation
PRICE_SPIKE_THRESHOLD = 0.20  # 20% spike in demand
OVERPRICED_THRESHOLD = 0.15  # 15% above median
UNDERPRICED_THRESHOLD = 0.10  # 10% below median


class PricingAgent:
    """Autonomous pricing agent for Vast.ai GPU hosts"""
    
    def __init__(self, api_key: str = VAST_API_KEY, monitor_interval: int = DEFAULT_MONITOR_INTERVAL):
        """
        Initialize the pricing agent
        
        Args:
            api_key: Vast.ai API key
            monitor_interval: Minutes between market checks
        """
        self.api_key = api_key
        self.monitor_interval = monitor_interval
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.market_history = []
        
        # Load the trained ML model
        self._load_model()
        
        print(f"[PricingAgent] Initialized with {monitor_interval}min monitoring interval")
    
    def _load_model(self):
        """Load the trained pricing model"""
        model_path = MODEL_DIR / "pricing_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        print(f"[PricingAgent] Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
        
        print(f"[PricingAgent] Model loaded successfully")
        print(f"[PricingAgent] Features: {self.feature_names}")
    
    def fetch_market_data(self) -> Dict:
        """
        Fetch live market data from Vast.ai API
        
        Returns:
            Dictionary containing market snapshot
        """
        print(f"\n[MarketMonitor] Fetching live market data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Query for verified, rentable GPU offers
        query = {
            "verified": {"eq": True},
            "rentable": {"eq": True}
        }
        
        url = f"{VAST_API_BASE}/bundles/"
        params = {"q": json.dumps(query)}
        headers = {"Accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            offers = response.json().get('offers', [])
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'total_offers': len(offers),
                'offers': offers
            }
            
            print(f"[MarketMonitor] Fetched {len(offers)} offers")
            
            return snapshot
            
        except requests.exceptions.RequestException as e:
            print(f"[MarketMonitor] ERROR: Failed to fetch market data: {e}")
            return None
    
    def analyze_market_trends(self, current_snapshot: Dict, previous_snapshot: Optional[Dict] = None) -> Dict:
        """
        Analyze market trends by comparing current vs previous snapshot
        
        Args:
            current_snapshot: Current market data
            previous_snapshot: Previous market data (if available)
        
        Returns:
            Dictionary with trend analysis
        """
        print(f"\n[MarketAnalyzer] Analyzing market trends...")
        
        current_df = pd.DataFrame(current_snapshot['offers'])
        
        # Basic statistics
        trends = {
            'timestamp': current_snapshot['timestamp'],
            'total_offers': len(current_df),
            'gpu_types': {},
            'demand_spike_detected': False,
            'price_drop_detected': False
        }
        
        # Analyze by GPU type
        for gpu_name in current_df['gpu_name'].unique():
            gpu_data = current_df[current_df['gpu_name'] == gpu_name]
            
            gpu_stats = {
                'count': len(gpu_data),
                'median_price': float(gpu_data['dph_base'].median()),
                'mean_price': float(gpu_data['dph_base'].mean()),
                'min_price': float(gpu_data['dph_base'].min()),
                'max_price': float(gpu_data['dph_base'].max()),
                'std_price': float(gpu_data['dph_base'].std()),
                'total_rentals': int(gpu_data.get('rentable', 1).sum())
            }
            
            trends['gpu_types'][gpu_name] = gpu_stats
        
        # Compare with previous snapshot if available
        if previous_snapshot and len(self.market_history) > 0:
            prev_df = pd.DataFrame(previous_snapshot['offers'])
            
            # Check for demand spikes (increase in rentals)
            current_rentals = current_df.get('rentable', pd.Series([0])).sum()
            prev_rentals = prev_df.get('rentable', pd.Series([0])).sum()
            
            if prev_rentals > 0:
                rental_change = (current_rentals - prev_rentals) / prev_rentals
                if rental_change > PRICE_SPIKE_THRESHOLD:
                    trends['demand_spike_detected'] = True
                    trends['demand_change_pct'] = float(rental_change * 100)
            
            # Check for price drops
            current_median = current_df['dph_base'].median()
            prev_median = prev_df['dph_base'].median()
            
            if prev_median > 0:
                price_change = (current_median - prev_median) / prev_median
                if price_change < -PRICE_DROP_THRESHOLD:
                    trends['price_drop_detected'] = True
                    trends['price_change_pct'] = float(price_change * 100)
        
        print(f"[MarketAnalyzer] Analyzed {len(trends['gpu_types'])} GPU types")
        if trends['demand_spike_detected']:
            print(f"[MarketAnalyzer] *** DEMAND SPIKE DETECTED: +{trends.get('demand_change_pct', 0):.1f}% ***")
        if trends['price_drop_detected']:
            print(f"[MarketAnalyzer] *** PRICE DROP DETECTED: {trends.get('price_change_pct', 0):.1f}% ***")
        
        return trends
    
    def predict_optimal_price(self, gpu_config: Dict) -> Tuple[float, float, float]:
        """
        Use ML model to predict optimal price for a GPU configuration
        
        Args:
            gpu_config: Dictionary with GPU configuration details
        
        Returns:
            Tuple of (predicted_price, lower_bound, upper_bound)
        """
        # Prepare features in the correct order
        feature_values = []
        for feature in self.feature_names:
            if feature == 'gpu_name_encoded':
                # For now, use a simple hash-based encoding
                # In production, would use the same LabelEncoder from training
                feature_values.append(hash(gpu_config.get('gpu_name', '')) % 100)
            else:
                feature_values.append(gpu_config.get(feature, 0))
        
        # Create feature array
        X = np.array([feature_values])
        
        # Scale if scaler is available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Calculate confidence interval (using model uncertainty if available)
        # For Random Forest, use std of tree predictions
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            std = tree_predictions.std()
            lower_bound = prediction - 1.96 * std
            upper_bound = prediction + 1.96 * std
        else:
            # Default 20% confidence interval
            lower_bound = prediction * 0.8
            upper_bound = prediction * 1.2
        
        return float(prediction), float(lower_bound), float(upper_bound)
    
    def generate_recommendations(self, market_snapshot: Dict, trends: Dict) -> List[Dict]:
        """
        Generate pricing recommendations based on market data and ML predictions
        
        Args:
            market_snapshot: Current market snapshot
            trends: Market trend analysis
        
        Returns:
            List of recommendation dictionaries
        """
        print(f"\n[DecisionEngine] Generating pricing recommendations...")
        
        recommendations = []
        df = pd.DataFrame(market_snapshot['offers'])
        
        # Process each GPU type
        for gpu_name, stats in trends['gpu_types'].items():
            gpu_offers = df[df['gpu_name'] == gpu_name]
            
            if len(gpu_offers) == 0:
                continue
            
            # Get representative configuration
            typical_config = gpu_offers.iloc[0].to_dict()
            
            # Get ML model prediction
            try:
                predicted_price, lower_bound, upper_bound = self.predict_optimal_price(typical_config)
            except Exception as e:
                print(f"[DecisionEngine] Warning: Could not predict for {gpu_name}: {e}")
                predicted_price = stats['median_price']
                lower_bound = predicted_price * 0.8
                upper_bound = predicted_price * 1.2
            
            # Current market conditions
            current_median = stats['median_price']
            current_mean = stats['mean_price']
            
            # Decision logic
            action = "hold"
            reason = []
            confidence = 0.5
            recommended_price = current_median
            
            # Rule 1: Market price drop detection
            if trends.get('price_drop_detected', False):
                if current_median < predicted_price * 0.9:
                    action = "decrease"
                    recommended_price = max(MIN_PRICE, current_median * 0.95)
                    reason.append(f"Market-wide price drop detected ({trends.get('price_change_pct', 0):.1f}%)")
                    confidence += 0.2
            
            # Rule 2: Demand spike detection
            if trends.get('demand_spike_detected', False):
                action = "increase"
                recommended_price = min(MAX_PRICE, current_median * 1.1)
                reason.append(f"Demand spike detected (+{trends.get('demand_change_pct', 0):.1f}%)")
                confidence += 0.3
            
            # Rule 3: Compare current vs ML prediction
            price_vs_prediction = (current_median - predicted_price) / predicted_price if predicted_price > 0 else 0
            
            if price_vs_prediction > OVERPRICED_THRESHOLD:
                if action != "increase":  # Don't override demand spike
                    action = "decrease"
                    recommended_price = max(MIN_PRICE, predicted_price)
                    reason.append(f"Current price {price_vs_prediction*100:.1f}% above ML prediction")
                    confidence += 0.25
            elif price_vs_prediction < -UNDERPRICED_THRESHOLD:
                action = "increase"
                recommended_price = min(MAX_PRICE, predicted_price)
                reason.append(f"Current price {abs(price_vs_prediction)*100:.1f}% below ML prediction (leaving money on table)")
                confidence += 0.25
            
            # Rule 4: Price position vs market
            if current_median > stats['mean_price'] * 1.15:
                if action != "increase":
                    action = "decrease"
                    recommended_price = max(MIN_PRICE, stats['mean_price'])
                    reason.append("Price significantly above market average")
                    confidence += 0.15
            
            # Default reason if holding
            if not reason:
                reason.append("Market conditions stable, no adjustment needed")
                recommended_price = current_median
            
            # Ensure guardrails
            recommended_price = max(MIN_PRICE, min(MAX_PRICE, recommended_price))
            
            # Calculate estimated impact
            if action == "increase":
                estimated_impact = f"+${(recommended_price - current_median):.3f}/hr, potential revenue gain but may reduce bookings"
            elif action == "decrease":
                estimated_impact = f"-${(current_median - recommended_price):.3f}/hr, increase competitiveness"
            else:
                estimated_impact = "No change, maintain current position"
            
            # Normalize confidence
            confidence = min(1.0, confidence)
            
            recommendation = {
                'gpu_type': gpu_name,
                'current_market_median': round(current_median, 4),
                'current_market_mean': round(current_mean, 4),
                'ml_predicted_price': round(predicted_price, 4),
                'ml_confidence_interval': {
                    'lower': round(lower_bound, 4),
                    'upper': round(upper_bound, 4)
                },
                'recommended_price': round(recommended_price, 4),
                'action': action,
                'reason': '; '.join(reason),
                'confidence': round(confidence, 2),
                'estimated_impact': estimated_impact,
                'market_sample_size': stats['count'],
                'timestamp': market_snapshot['timestamp']
            }
            
            recommendations.append(recommendation)
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"[DecisionEngine] Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def save_recommendations(self, recommendations: List[Dict]):
        """Save recommendations to JSON file"""
        output_path = DATA_DIR / "agent_recommendations.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'recommendations': recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[Agent] Saved {len(recommendations)} recommendations to {output_path}")
    
    def log_run(self, recommendations: List[Dict], market_snapshot: Dict, trends: Dict):
        """Append run log to JSONL file"""
        log_path = LOGS_DIR / "agent_run_log.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'market_total_offers': market_snapshot['total_offers'],
            'trends': {
                'demand_spike_detected': trends.get('demand_spike_detected', False),
                'price_drop_detected': trends.get('price_drop_detected', False),
                'gpu_types_analyzed': len(trends['gpu_types'])
            },
            'recommendations': {
                'total': len(recommendations),
                'increase': sum(1 for r in recommendations if r['action'] == 'increase'),
                'decrease': sum(1 for r in recommendations if r['action'] == 'decrease'),
                'hold': sum(1 for r in recommendations if r['action'] == 'hold'),
                'high_confidence': sum(1 for r in recommendations if r['confidence'] >= 0.7)
            }
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"[Agent] Logged run to {log_path}")
    
    def print_summary(self, recommendations: List[Dict]):
        """Print a formatted summary of recommendations"""
        print("\n" + "="*100)
        print("PRICING AGENT RECOMMENDATIONS SUMMARY")
        print("="*100)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Recommendations: {len(recommendations)}")
        print()
        
        # Action summary
        actions = {'increase': 0, 'decrease': 0, 'hold': 0}
        for rec in recommendations:
            actions[rec['action']] += 1
        
        print(f"Actions: INCREASE={actions['increase']} | DECREASE={actions['decrease']} | HOLD={actions['hold']}")
        print("\n" + "-"*100)
        
        # Top recommendations (high confidence)
        high_confidence = [r for r in recommendations if r['confidence'] >= 0.7]
        if high_confidence:
            print(f"\nHIGH CONFIDENCE RECOMMENDATIONS ({len(high_confidence)}):")
            print("-"*100)
            
            for rec in high_confidence[:10]:  # Top 10
                print(f"\nGPU: {rec['gpu_type']}")
                print(f"  Current Market Median: ${rec['current_market_median']:.4f}/hr")
                print(f"  ML Predicted Price: ${rec['ml_predicted_price']:.4f}/hr (CI: ${rec['ml_confidence_interval']['lower']:.4f} - ${rec['ml_confidence_interval']['upper']:.4f})")
                print(f"  >>> RECOMMENDED: ${rec['recommended_price']:.4f}/hr | ACTION: {rec['action'].upper()} | Confidence: {rec['confidence']:.0%}")
                print(f"  Reason: {rec['reason']}")
                print(f"  Impact: {rec['estimated_impact']}")
                print(f"  Sample Size: {rec['market_sample_size']} offers")
        
        # All other recommendations
        medium_low = [r for r in recommendations if r['confidence'] < 0.7]
        if medium_low:
            print(f"\n\nOTHER RECOMMENDATIONS ({len(medium_low)}):")
            print("-"*100)
            
            for rec in medium_low[:15]:  # Show up to 15
                print(f"{rec['gpu_type']:30s} | Market: ${rec['current_market_median']:6.4f} | Predicted: ${rec['ml_predicted_price']:6.4f} | "
                      f"Recommend: ${rec['recommended_price']:6.4f} | {rec['action'].upper():8s} | Conf: {rec['confidence']:.0%}")
        
        print("\n" + "="*100)
    
    def run_cycle(self):
        """
        Run a complete monitoring and recommendation cycle
        
        Returns:
            List of recommendations
        """
        print("\n" + "="*100)
        print(f"PRICING AGENT CYCLE START - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Step 1: Fetch market data
        market_snapshot = self.fetch_market_data()
        if not market_snapshot:
            print("[Agent] ERROR: Failed to fetch market data, aborting cycle")
            return None
        
        # Step 2: Analyze trends (compare with previous if available)
        previous_snapshot = self.market_history[-1] if self.market_history else None
        trends = self.analyze_market_trends(market_snapshot, previous_snapshot)
        
        # Store in history
        self.market_history.append(market_snapshot)
        if len(self.market_history) > 10:  # Keep last 10 snapshots
            self.market_history.pop(0)
        
        # Step 3: Generate recommendations
        recommendations = self.generate_recommendations(market_snapshot, trends)
        
        # Step 4: Save outputs
        self.save_recommendations(recommendations)
        self.log_run(recommendations, market_snapshot, trends)
        
        # Step 5: Print summary
        self.print_summary(recommendations)
        
        print("\n" + "="*100)
        print(f"PRICING AGENT CYCLE COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        return recommendations
    
    def run_continuous(self, max_cycles: Optional[int] = None):
        """
        Run the agent continuously with periodic monitoring
        
        Args:
            max_cycles: Maximum number of cycles (None = infinite)
        """
        cycle_count = 0
        
        print(f"[Agent] Starting continuous monitoring (interval: {self.monitor_interval} minutes)")
        
        try:
            while max_cycles is None or cycle_count < max_cycles:
                self.run_cycle()
                
                cycle_count += 1
                
                if max_cycles is None or cycle_count < max_cycles:
                    print(f"[Agent] Waiting {self.monitor_interval} minutes until next cycle...")
                    time.sleep(self.monitor_interval * 60)
        
        except KeyboardInterrupt:
            print("\n[Agent] Stopped by user")
        
        print(f"[Agent] Completed {cycle_count} cycles")


def main():
    """Demo: Run a single pricing agent cycle"""
    print("Vast.ai Autonomous Pricing Agent - Demo Run")
    print("=" * 100)
    
    # Initialize agent
    agent = PricingAgent(
        api_key=VAST_API_KEY,
        monitor_interval=15
    )
    
    # Run one cycle
    recommendations = agent.run_cycle()
    
    if recommendations:
        print(f"\n[Demo] Success! Generated {len(recommendations)} pricing recommendations")
        print(f"[Demo] Results saved to: {DATA_DIR / 'agent_recommendations.json'}")
        print(f"[Demo] Run log saved to: {LOGS_DIR / 'agent_run_log.jsonl'}")
    else:
        print("\n[Demo] Failed to generate recommendations")
    
    return recommendations


if __name__ == "__main__":
    main()
