"""
LLM Explainer for Vast.ai Pricing Agent
Uses Chutes.ai API to generate plain English explanations for pricing decisions.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import httpx


class PricingExplainer:
    """Generates human-readable explanations for ML pricing recommendations using LLM."""
    
    def __init__(
        self,
        api_key: str = "cpk_642cdd7eb4624de4840a0c9cc3e724c3.db4e7a98d9eb62b374a5f4fec19cb3ae.mNyutVCzXBM9JHCQEAhRBkSbRqaS3Sas",
        base_url: str = "https://llm.chutes.ai/v1",
        model: str = "deepseek-ai/DeepSeek-V3-0324",
        fallback_model: str = "unsloth/Llama-3.3-70B-Instruct"
    ):
        """
        Initialize the PricingExplainer.
        
        Args:
            api_key: Chutes.ai API key
            base_url: Chutes.ai base URL
            model: Primary LLM model to use
            fallback_model: Fallback model if primary fails
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.fallback_model = fallback_model
        self.client = httpx.Client(timeout=30.0)
        
    def _call_llm(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
        """
        Call Chutes.ai LLM API with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLM response text or None if all attempts fail
        """
        models_to_try = [self.model, self.fallback_model]
        
        # Try both Bearer and X-API-Key authentication methods
        auth_methods = [
            {"Authorization": f"Bearer {self.api_key}"},
            {"X-API-Key": self.api_key}
        ]
        
        for model in models_to_try:
            for auth_header in auth_methods:
                try:
                    headers = {**auth_header, "Content-Type": "application/json"}
                    response = self.client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a GPU pricing analyst explaining machine learning pricing decisions. Be concise, specific, and use actual numbers from the data. Write 2-3 sentences maximum."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"].strip()
                    # Only print error on last auth attempt
                    elif auth_header == auth_methods[-1]:
                        print(f"LLM API error with {model}: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    # Only print error on last auth attempt
                    if auth_header == auth_methods[-1]:
                        print(f"LLM call failed with {model}: {str(e)}")
                    continue
        
        return None
    
    def explain_recommendation(self, recommendation: Dict) -> str:
        """
        Generate plain English explanation for a single pricing recommendation.
        
        Args:
            recommendation: A recommendation dict from agent_recommendations.json
            
        Returns:
            2-3 sentence explanation of the pricing decision
        """
        gpu_type = recommendation["gpu_type"]
        current_median = recommendation["current_market_median"]
        ml_price = recommendation["ml_predicted_price"]
        action = recommendation["action"]
        reason = recommendation["reason"]
        confidence = recommendation["confidence"]
        sample_size = recommendation["market_sample_size"]
        impact = recommendation["estimated_impact"]
        
        # Construct prompt with all relevant context
        prompt = f"""Explain this GPU pricing recommendation in 2-3 sentences:

GPU: {gpu_type}
Current Market Median: ${current_median:.2f}/hr
ML Model Recommendation: ${ml_price:.2f}/hr
Action: {action.upper()}
Reason: {reason}
Confidence: {confidence:.0%}
Market Sample Size: {sample_size} offers
Estimated Impact: {impact}

Write a concise explanation that a business executive would understand, focusing on WHY this price change makes strategic sense based on market conditions."""

        explanation = self._call_llm(prompt, max_tokens=150, temperature=0.7)
        
        # Fallback to rule-based explanation if LLM fails
        if not explanation:
            if action == "decrease":
                explanation = f"The {gpu_type} is currently priced {reason.split('%')[0].split()[-1]}% above our ML model's prediction of ${ml_price:.2f}/hr. Reducing from the market median of ${current_median:.2f}/hr will improve competitiveness and increase rental probability in a market with {sample_size} competing offers."
            elif action == "increase":
                explanation = f"The {gpu_type} shows strong demand with our ML model recommending ${ml_price:.2f}/hr, which is {reason.split('%')[0].split()[-1]}% above the current market median of ${current_median:.2f}/hr. This price increase captures additional revenue while maintaining competitive positioning."
            else:
                explanation = f"The {gpu_type} is optimally priced at ${current_median:.2f}/hr based on current market conditions. Our ML model recommends holding this price point with {confidence:.0%} confidence across {sample_size} market samples."
        
        return explanation
    
    def explain_all_recommendations(
        self,
        recommendations_file: str = "code/vastai-pricing/data/agent_recommendations.json",
        top_n: int = 5
    ) -> List[Dict]:
        """
        Generate explanations for top N recommendations.
        
        Args:
            recommendations_file: Path to agent recommendations JSON
            top_n: Number of top recommendations to explain
            
        Returns:
            List of dicts with recommendation + LLM explanation
        """
        try:
            with open(recommendations_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: {recommendations_file} not found")
            return []
        
        recommendations = data.get("recommendations", [])[:top_n]
        explained = []
        
        print(f"\nGenerating LLM explanations for top {len(recommendations)} recommendations...")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  [{i}/{len(recommendations)}] Explaining {rec['gpu_type']}...")
            
            explanation = self.explain_recommendation(rec)
            
            explained.append({
                "gpu_type": rec["gpu_type"],
                "action": rec["action"],
                "current_price": rec["current_market_median"],
                "recommended_price": rec["recommended_price"],
                "confidence": rec["confidence"],
                "llm_explanation": explanation
            })
        
        print("Done!")
        return explained
    
    def generate_market_summary(
        self,
        snapshot_file: str = "code/vastai-pricing/data/market_snapshot.json",
        recommendations_file: str = "code/vastai-pricing/data/agent_recommendations.json"
    ) -> str:
        """
        Generate executive summary of current GPU market conditions.
        
        Args:
            snapshot_file: Path to market snapshot JSON
            recommendations_file: Path to agent recommendations JSON
            
        Returns:
            3-4 sentence executive summary suitable for dashboard header
        """
        try:
            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
            with open(recommendations_file, 'r') as f:
                recs = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading market data: {e}")
            return "Market data unavailable."
        
        total_offers = snapshot.get("total_offers", 0)
        total_recs = recs.get("total_recommendations", 0)
        recommendations = recs.get("recommendations", [])
        
        # Calculate action distribution
        actions = {}
        for rec in recommendations:
            action = rec["action"]
            actions[action] = actions.get(action, 0) + 1
        
        # Get sample of GPU types
        gpu_types = list(set([rec["gpu_type"] for rec in recommendations[:10]]))
        
        # Get average market prices
        market_prices = [rec["current_market_median"] for rec in recommendations]
        avg_market_price = sum(market_prices) / len(market_prices) if market_prices else 0
        
        prompt = f"""Write a 3-4 sentence executive summary of the current GPU rental market:

Total Market Offers: {total_offers}
GPU Types Analyzed: {total_recs}
Sample GPUs: {', '.join(gpu_types[:5])}
Average Market Price: ${avg_market_price:.2f}/hr
Recommended Actions: {actions.get('decrease', 0)} decrease, {actions.get('increase', 0)} increase, {actions.get('hold', 0)} hold

Write this for a business dashboard header - focus on market trends, pricing opportunities, and strategic implications."""

        summary = self._call_llm(prompt, max_tokens=250, temperature=0.7)
        
        # Fallback to rule-based summary
        if not summary:
            summary = f"Vast.ai market analysis covers {total_offers} active GPU offers across {total_recs} GPU types. Our ML pricing model recommends {actions.get('decrease', 0)} price decreases to improve competitiveness, {actions.get('increase', 0)} increases to capture demand, and {actions.get('hold', 0)} holds for optimal positioning. Average market price is ${avg_market_price:.2f}/hr with opportunities in {', '.join(gpu_types[:3])} segments."
        
        return summary
    
    def close(self):
        """Close HTTP client."""
        self.client.close()


def main():
    """Run live demo of LLM explainer."""
    print("=" * 80)
    print("Vast.ai Pricing Agent - LLM Explainer Demo")
    print("=" * 80)
    
    # Initialize explainer
    explainer = PricingExplainer()
    
    try:
        # Generate market narrative
        print("\n[1/2] Generating Market Narrative...")
        print("-" * 80)
        market_summary = explainer.generate_market_summary()
        print(f"\n{market_summary}\n")
        
        # Generate explanations for top 5 recommendations
        print("\n[2/2] Generating Pricing Explanations...")
        print("-" * 80)
        explained_recommendations = explainer.explain_all_recommendations(top_n=5)
        
        # Print all explanations
        print("\n" + "=" * 80)
        print("PRICING RECOMMENDATIONS WITH LLM EXPLANATIONS")
        print("=" * 80)
        
        for i, rec in enumerate(explained_recommendations, 1):
            print(f"\n[{i}] {rec['gpu_type']}")
            print(f"    Action: {rec['action'].upper()}")
            print(f"    Current: ${rec['current_price']:.2f}/hr")
            print(f"    Recommended: ${rec['recommended_price']:.2f}/hr")
            print(f"    Confidence: {rec['confidence']:.0%}")
            print(f"\n    Explanation:")
            print(f"    {rec['llm_explanation']}")
        
        # Save all to JSON
        output_file = "code/vastai-pricing/data/llm_explanations.json"
        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "market_summary": market_summary,
            "explained_recommendations": explained_recommendations
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_file}")
        print("=" * 80)
        
    finally:
        explainer.close()


if __name__ == "__main__":
    main()
