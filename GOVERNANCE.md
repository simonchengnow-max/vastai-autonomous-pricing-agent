# Vast.ai AI Pricing Agent - Governance Layer

## Overview

The governance layer provides enterprise-grade safety controls, transparency, and accountability for autonomous pricing decisions. It ensures all price changes pass through validation gates before being applied to production.

## Key Components

### 1. Guardrails (Safety Constraints)

Eight automated guardrails protect against dangerous pricing decisions:

| Guardrail | Description | Severity | Default Threshold |
|-----------|-------------|----------|-------------------|
| **Price Floor** | Minimum price limit | Critical | $0.10/hr |
| **Price Ceiling** | Maximum price limit | Critical | $20.00/hr |
| **Max Increase** | Limit price increase % | High | 50% |
| **Max Decrease** | Limit price decrease % | High | 50% |
| **Market Sample** | Minimum data points required | Medium | 3 offers |
| **Confidence** | ML model confidence threshold | Medium | 50% (propose), 80% (auto-apply) |
| **Market Deviation** | Max deviation from market median | High | 75% |
| **Price Sanity** | Basic validity check | Critical | >$0, <$1000, not NaN |

### 2. Risk Assessment

Every pricing decision is classified into risk levels:

- **LOW**: Passes all guardrails, high confidence (≥80%), sufficient market data
- **MEDIUM**: Low confidence or small market sample, but within guardrails
- **HIGH**: Failed high-severity guardrail or large price change (>30%)
- **CRITICAL**: Failed critical guardrail (price floor/ceiling/sanity)

### 3. Decision Pipeline

```
Recommendation → Validation → Risk Assessment → Approval → Application
     (ML)           (All)      (Classification)  (Human/Auto)  (Vast.ai API)
```

**Status Flow:**
1. **PROPOSED** - Initial ML recommendation
2. **VALIDATED** - Passed all guardrails
3. **APPROVED** - Approved for application (auto or human)
4. **REJECTED** - Failed critical guardrails
5. **APPLIED** - Successfully applied to Vast.ai
6. **ROLLED_BACK** - Reverted due to issues

### 4. Audit Trail

Comprehensive logging of every decision with:
- Unique decision ID (SHA-256 hash)
- GPU type, current/recommended price
- Action (increase/decrease/hold)
- Risk level and confidence score
- All guardrails passed/failed
- Validator (ML/LLM/Human)
- Timestamps and validation notes
- Application status and rollback references

**Storage:**
- `logs/governance_audit.jsonl` - Append-only audit log
- `data/decision_history.json` - Per-GPU decision history (last 50)
- `data/governance_report.json` - Summary statistics

## Usage

### Basic Validation

```python
from governance import PricingGovernance

# Initialize governance system
governance = PricingGovernance()

# Validate a recommendation
recommendation = {
    'gpu_type': 'RTX 4090',
    'current_market_median': 1.50,
    'recommended_price': 1.80,
    'action': 'increase',
    'confidence': 0.85,
    'market_sample_size': 5
}

passed, guardrails_passed, guardrails_failed, risk_level = governance.validate_decision(recommendation)

if passed:
    print(f"✅ Validated - Risk: {risk_level.value}")
else:
    print(f"❌ Rejected - Failed: {guardrails_failed}")
```

### Creating Audit Entries

```python
from governance import DecisionStatus

# Create audit entry
entry = governance.create_audit_entry(
    recommendation,
    guardrails_passed,
    guardrails_failed,
    risk_level,
    DecisionStatus.VALIDATED,
    validator="ML+Governance",
    validation_notes="Automated validation"
)

# Approve low-risk decisions
if risk_level == RiskLevel.LOW:
    governance.approve_decision(entry.decision_id, "AutoApproval")
    governance.apply_decision(entry.decision_id)
```

### Rollback

```python
# Rollback a problematic decision
rollback_id = governance.rollback_decision(
    decision_id="abc123def456",
    reason="Customer complaint - price too high"
)
```

### Governance Report

```python
report = governance.get_governance_report()
print(f"Total Decisions: {report['total_decisions']}")
print(f"Applied: {report['applied_decisions']}")
print(f"Rollbacks: {report['rollback_count']}")
print(f"Avg Confidence: {report['avg_confidence']:.1%}")
```

## Current Governance Results

Based on the latest run:

```
Total Decisions Processed: 22
✅ Validated: 8 (36%)
❌ Rejected: 14 (64%)
⚠️  High/Critical Risk: 17 (77%)
📊 Average Confidence: 72.7%
```

**Top Guardrail Violations:**
1. Market Sample Check (12) - Insufficient market data
2. Max Increase Check (9) - Price increase >50%
3. Market Deviation Check (6) - Too far from median
4. Max Decrease Check (5) - Price decrease >50%

## Configuration

Customize governance rules via `GovernanceConfig`:

```python
from governance import GovernanceConfig, PricingGovernance

config = GovernanceConfig(
    max_price_increase_pct=30.0,  # Stricter: 30% max increase
    max_price_decrease_pct=30.0,  # Stricter: 30% max decrease
    min_confidence_auto_apply=0.90,  # Higher confidence required
    min_market_sample_size=5,  # More data required
    max_deviation_from_median=0.50,  # Stay closer to market
)

governance = PricingGovernance(config)
```

## Integration with Pricing Agent

The governance layer integrates with the pricing agent through:

1. **pricing_agent.py** generates recommendations
2. **governance.py** validates and creates audit trail
3. **dashboard.html** visualizes decisions and guardrail status
4. Human oversight for high-risk decisions before Vast.ai API calls

## Best Practices

### For Autonomous Operation

1. **Start Conservative** - Use stricter thresholds (30% max changes, 90% confidence)
2. **Monitor Closely** - Review governance reports daily for first 2 weeks
3. **Auto-Apply Low Risk Only** - Require human approval for medium/high risk
4. **Track Rollbacks** - If >10% rollback rate, tighten guardrails

### For Manual Override

1. **Document All Overrides** - Use validation_notes field
2. **Review Rejected Decisions** - High-value GPUs may need custom rules
3. **Adjust Guardrails** - Based on actual outcomes, not just ML predictions

### For Production Deployment

1. **Dry-Run Mode** - Log decisions without applying for 1 week
2. **Gradual Rollout** - Start with 1-2 GPU types, expand based on results
3. **Alert on Critical** - Notify humans immediately for critical risk decisions
4. **Daily Reports** - Email governance summary to stakeholders

## Files Generated

- `code/vastai-pricing/governance.py` - Main governance system
- `code/vastai-pricing/logs/governance_audit.jsonl` - Audit trail (append-only)
- `code/vastai-pricing/data/decision_history.json` - Per-GPU history
- `code/vastai-pricing/data/governance_report.json` - Summary statistics
- `code/vastai-pricing/GOVERNANCE.md` - This documentation

## Safety Features

### Fail-Safe Mechanisms

1. **Critical Guardrails** - Cannot be disabled, always enforced
2. **Audit Immutability** - Append-only logs prevent tampering
3. **Rollback Trail** - Every rollback references original decision
4. **Hash-Based IDs** - Decision IDs are cryptographic hashes

### Transparency

1. **Full Decision History** - Every recommendation logged with reasoning
2. **Guardrail Visibility** - See exactly why decisions failed
3. **Risk Classification** - Clear risk levels for human oversight
4. **Validation Notes** - Human reviewers can add context

### Compliance

The governance system supports:
- **Audit Requirements** - Full trail of all pricing decisions
- **Rollback Capability** - Undo any change with reason tracking
- **Human Oversight** - Manual approval gates for high-risk changes
- **Configuration Control** - Version-controlled governance rules

## Future Enhancements

1. **A/B Testing** - Track applied vs rejected recommendations
2. **Outcome Tracking** - Measure booking rate after price changes
3. **Dynamic Thresholds** - Adjust guardrails based on performance
4. **Alert Integration** - Slack/email notifications for critical decisions
5. **Multi-Level Approval** - Different approval levels for different risk tiers
6. **ML Guardrails** - Train model to predict decision success
7. **Competitor Benchmarking** - Guardrails based on competitive positioning

## Summary

The governance layer transforms the autonomous pricing agent from a pure ML system into an enterprise-grade decision support system with:

✅ **Safety** - 8 automated guardrails prevent dangerous price changes  
✅ **Transparency** - Comprehensive audit trail of all decisions  
✅ **Accountability** - Clear risk levels and approval workflows  
✅ **Control** - Human oversight for high-risk decisions  
✅ **Compliance** - Immutable logs and rollback capability  

The system successfully validated 8 of 22 recommendations (36%), automatically rejecting 14 high-risk decisions that violated safety constraints - exactly as designed.
