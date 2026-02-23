"""
Governance Layer for Vast.ai AI Pricing Agent

This module provides:
1. Guardrails - safety constraints on price changes
2. Audit Trail - comprehensive logging of all decisions
3. Decision Validation - multi-stage approval process
4. Risk Assessment - identify high-risk pricing decisions
5. Rollback Capabilities - undo problematic changes

Author: AI Pricing Agent Team
Date: 2026-02-23
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class RiskLevel(Enum):
    """Risk classification for pricing decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatus(Enum):
    """Status of pricing decisions through approval pipeline"""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass
class Guardrail:
    """Safety constraint for pricing decisions"""
    name: str
    description: str
    check_function: str  # Name of the validation function
    max_violation_severity: str  # low, medium, high, critical
    enabled: bool = True


@dataclass
class AuditEntry:
    """Comprehensive audit log entry"""
    timestamp: str
    decision_id: str
    gpu_type: str
    action: str
    current_price: float
    recommended_price: float
    change_percentage: float
    confidence: float
    risk_level: str
    status: str
    guardrails_passed: List[str]
    guardrails_failed: List[str]
    validator: str  # ML, LLM, Human
    validation_notes: str
    applied: bool
    rollback_id: Optional[str] = None


@dataclass
class GovernanceConfig:
    """Configuration for governance rules"""
    # Price change limits
    max_price_increase_pct: float = 50.0  # Max 50% increase
    max_price_decrease_pct: float = 50.0  # Max 50% decrease
    min_price_floor: float = 0.10  # Never go below $0.10/hr
    max_price_ceiling: float = 20.0  # Never exceed $20/hr
    
    # Confidence thresholds
    min_confidence_auto_apply: float = 0.80  # Auto-apply only if >= 80% confidence
    min_confidence_propose: float = 0.50  # Propose if >= 50% confidence
    
    # Market validation
    min_market_sample_size: int = 3  # Need at least 3 market samples
    max_deviation_from_median: float = 0.75  # Max 75% deviation from market median
    
    # Risk assessment
    high_value_gpu_threshold: float = 5.0  # GPUs > $5/hr are high-value
    critical_change_threshold: float = 30.0  # Changes > 30% are critical
    
    # Audit settings
    audit_log_path: str = "code/vastai-pricing/logs/governance_audit.jsonl"
    decision_history_path: str = "code/vastai-pricing/data/decision_history.json"
    max_audit_entries: int = 10000


class PricingGovernance:
    """
    Governance system for autonomous pricing decisions
    
    Ensures all pricing changes pass through validation gates,
    maintains comprehensive audit trail, and provides rollback capability.
    """
    
    def __init__(self, config: Optional[GovernanceConfig] = None):
        self.config = config or GovernanceConfig()
        self.audit_log: List[AuditEntry] = []
        self.decision_history: Dict[str, List[AuditEntry]] = {}
        
        # Initialize guardrails
        self.guardrails = self._initialize_guardrails()
        
        # Load existing audit trail
        self._load_audit_trail()
    
    def _initialize_guardrails(self) -> List[Guardrail]:
        """Define all safety guardrails"""
        return [
            Guardrail(
                name="price_floor_check",
                description="Ensure price never goes below minimum floor",
                check_function="check_price_floor",
                max_violation_severity="critical"
            ),
            Guardrail(
                name="price_ceiling_check",
                description="Ensure price never exceeds maximum ceiling",
                check_function="check_price_ceiling",
                max_violation_severity="critical"
            ),
            Guardrail(
                name="max_increase_check",
                description="Limit maximum percentage price increase",
                check_function="check_max_increase",
                max_violation_severity="high"
            ),
            Guardrail(
                name="max_decrease_check",
                description="Limit maximum percentage price decrease",
                check_function="check_max_decrease",
                max_violation_severity="high"
            ),
            Guardrail(
                name="market_sample_check",
                description="Ensure sufficient market data for decision",
                check_function="check_market_sample",
                max_violation_severity="medium"
            ),
            Guardrail(
                name="confidence_check",
                description="Ensure ML model confidence meets threshold",
                check_function="check_confidence",
                max_violation_severity="medium"
            ),
            Guardrail(
                name="market_deviation_check",
                description="Prevent extreme deviation from market median",
                check_function="check_market_deviation",
                max_violation_severity="high"
            ),
            Guardrail(
                name="price_sanity_check",
                description="Verify recommended price is reasonable",
                check_function="check_price_sanity",
                max_violation_severity="critical"
            ),
        ]
    
    def validate_decision(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str], List[str], RiskLevel]:
        """
        Validate a pricing recommendation through all guardrails
        
        Returns:
            (passed, guardrails_passed, guardrails_failed, risk_level)
        """
        passed_checks = []
        failed_checks = []
        
        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue
            
            check_method = getattr(self, guardrail.check_function)
            passed = check_method(recommendation)
            
            if passed:
                passed_checks.append(guardrail.name)
            else:
                failed_checks.append(f"{guardrail.name} ({guardrail.max_violation_severity})")
        
        # Determine overall pass/fail
        critical_failures = [f for f in failed_checks if "critical" in f]
        high_failures = [f for f in failed_checks if "high" in f]
        
        overall_pass = len(critical_failures) == 0 and len(high_failures) == 0
        
        # Assess risk level
        risk_level = self._assess_risk(recommendation, failed_checks)
        
        return overall_pass, passed_checks, failed_checks, risk_level
    
    def _assess_risk(self, recommendation: Dict[str, Any], failed_checks: List[str]) -> RiskLevel:
        """Assess overall risk level of a pricing decision"""
        # Critical if any critical guardrail failed
        if any("critical" in f for f in failed_checks):
            return RiskLevel.CRITICAL
        
        # High risk if high guardrail failed or large price change
        change_pct = abs(
            (recommendation['recommended_price'] - recommendation['current_market_median']) 
            / recommendation['current_market_median'] * 100
        )
        
        if any("high" in f for f in failed_checks) or change_pct > self.config.critical_change_threshold:
            return RiskLevel.HIGH
        
        # Medium risk if confidence is low or market sample is small
        if (recommendation.get('confidence', 1.0) < self.config.min_confidence_auto_apply or
            recommendation.get('market_sample_size', 0) < self.config.min_market_sample_size):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    # Guardrail check functions
    def check_price_floor(self, rec: Dict[str, Any]) -> bool:
        """Ensure recommended price meets minimum floor"""
        return rec['recommended_price'] >= self.config.min_price_floor
    
    def check_price_ceiling(self, rec: Dict[str, Any]) -> bool:
        """Ensure recommended price doesn't exceed ceiling"""
        return rec['recommended_price'] <= self.config.max_price_ceiling
    
    def check_max_increase(self, rec: Dict[str, Any]) -> bool:
        """Check price increase doesn't exceed maximum"""
        if rec['action'] != 'increase':
            return True
        
        increase_pct = (rec['recommended_price'] - rec['current_market_median']) / rec['current_market_median'] * 100
        return increase_pct <= self.config.max_price_increase_pct
    
    def check_max_decrease(self, rec: Dict[str, Any]) -> bool:
        """Check price decrease doesn't exceed maximum"""
        if rec['action'] != 'decrease':
            return True
        
        decrease_pct = (rec['current_market_median'] - rec['recommended_price']) / rec['current_market_median'] * 100
        return decrease_pct <= self.config.max_price_decrease_pct
    
    def check_market_sample(self, rec: Dict[str, Any]) -> bool:
        """Ensure sufficient market sample size"""
        return rec.get('market_sample_size', 0) >= self.config.min_market_sample_size
    
    def check_confidence(self, rec: Dict[str, Any]) -> bool:
        """Check ML model confidence meets threshold"""
        return rec.get('confidence', 0) >= self.config.min_confidence_propose
    
    def check_market_deviation(self, rec: Dict[str, Any]) -> bool:
        """Check recommended price isn't too far from market median"""
        deviation = abs(rec['recommended_price'] - rec['current_market_median']) / rec['current_market_median']
        return deviation <= self.config.max_deviation_from_median
    
    def check_price_sanity(self, rec: Dict[str, Any]) -> bool:
        """Basic sanity check on recommended price"""
        price = rec['recommended_price']
        return price > 0 and price < 1000 and not (price != price)  # Not NaN
    
    def create_audit_entry(
        self, 
        recommendation: Dict[str, Any],
        guardrails_passed: List[str],
        guardrails_failed: List[str],
        risk_level: RiskLevel,
        status: DecisionStatus,
        validator: str = "ML",
        validation_notes: str = ""
    ) -> AuditEntry:
        """Create comprehensive audit trail entry"""
        
        decision_id = self._generate_decision_id(recommendation)
        change_pct = (
            (recommendation['recommended_price'] - recommendation['current_market_median']) 
            / recommendation['current_market_median'] * 100
        )
        
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            decision_id=decision_id,
            gpu_type=recommendation['gpu_type'],
            action=recommendation['action'],
            current_price=recommendation['current_market_median'],
            recommended_price=recommendation['recommended_price'],
            change_percentage=change_pct,
            confidence=recommendation.get('confidence', 0),
            risk_level=risk_level.value,
            status=status.value,
            guardrails_passed=guardrails_passed,
            guardrails_failed=guardrails_failed,
            validator=validator,
            validation_notes=validation_notes,
            applied=False
        )
        
        self.audit_log.append(entry)
        
        # Track by GPU type
        if entry.gpu_type not in self.decision_history:
            self.decision_history[entry.gpu_type] = []
        self.decision_history[entry.gpu_type].append(entry)
        
        return entry
    
    def _generate_decision_id(self, recommendation: Dict[str, Any]) -> str:
        """Generate unique decision ID"""
        data = f"{recommendation['gpu_type']}_{recommendation['timestamp']}_{recommendation['recommended_price']}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def approve_decision(self, decision_id: str, approver: str = "System") -> bool:
        """Approve a validated decision for application"""
        for entry in self.audit_log:
            if entry.decision_id == decision_id and entry.status == DecisionStatus.VALIDATED.value:
                entry.status = DecisionStatus.APPROVED.value
                entry.validator = approver
                entry.validation_notes += f" | Approved by {approver}"
                self._save_audit_trail()
                return True
        return False
    
    def apply_decision(self, decision_id: str) -> bool:
        """Mark decision as applied (would integrate with Vast.ai API)"""
        for entry in self.audit_log:
            if entry.decision_id == decision_id and entry.status == DecisionStatus.APPROVED.value:
                entry.status = DecisionStatus.APPLIED.value
                entry.applied = True
                self._save_audit_trail()
                return True
        return False
    
    def rollback_decision(self, decision_id: str, reason: str) -> Optional[str]:
        """Rollback an applied decision"""
        for entry in self.audit_log:
            if entry.decision_id == decision_id and entry.applied:
                rollback_id = f"rb_{decision_id}"
                entry.status = DecisionStatus.ROLLED_BACK.value
                entry.rollback_id = rollback_id
                entry.validation_notes += f" | ROLLBACK: {reason}"
                self._save_audit_trail()
                return rollback_id
        return None
    
    def get_governance_report(self) -> Dict[str, Any]:
        """Generate comprehensive governance report"""
        total_decisions = len(self.audit_log)
        
        if total_decisions == 0:
            return {"message": "No decisions logged yet"}
        
        status_counts = {}
        risk_counts = {}
        action_counts = {}
        
        for entry in self.audit_log:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
            risk_counts[entry.risk_level] = risk_counts.get(entry.risk_level, 0) + 1
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
        
        # Recent high-risk decisions
        high_risk = [
            {
                "decision_id": e.decision_id,
                "gpu_type": e.gpu_type,
                "risk_level": e.risk_level,
                "status": e.status,
                "failed_guardrails": e.guardrails_failed
            }
            for e in self.audit_log[-20:]
            if e.risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        ]
        
        # Guardrail violation summary
        all_violations = {}
        for entry in self.audit_log:
            for violation in entry.guardrails_failed:
                all_violations[violation] = all_violations.get(violation, 0) + 1
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_decisions": total_decisions,
            "status_breakdown": status_counts,
            "risk_breakdown": risk_counts,
            "action_breakdown": action_counts,
            "recent_high_risk_decisions": high_risk[-10:],
            "guardrail_violations": all_violations,
            "applied_decisions": sum(1 for e in self.audit_log if e.applied),
            "rollback_count": sum(1 for e in self.audit_log if e.rollback_id is not None),
            "avg_confidence": sum(e.confidence for e in self.audit_log) / total_decisions,
        }
    
    def _save_audit_trail(self):
        """Save audit trail to disk"""
        log_path = Path(self.config.audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        with open(log_path, 'a') as f:
            for entry in self.audit_log[-100:]:  # Save last 100 entries
                f.write(json.dumps(asdict(entry)) + '\n')
        
        # Save decision history
        history_path = Path(self.config.decision_history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_serializable = {
            gpu: [asdict(e) for e in entries[-50:]]  # Keep last 50 per GPU
            for gpu, entries in self.decision_history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def _load_audit_trail(self):
        """Load existing audit trail"""
        log_path = Path(self.config.audit_log_path)
        if log_path.exists():
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entry = AuditEntry(**data)
                        self.audit_log.append(entry)
                        
                        if entry.gpu_type not in self.decision_history:
                            self.decision_history[entry.gpu_type] = []
                        self.decision_history[entry.gpu_type].append(entry)
                    except Exception as e:
                        print(f"Error loading audit entry: {e}")


def main():
    """Demo: Run governance validation on existing recommendations"""
    
    # Load existing recommendations
    with open('code/vastai-pricing/data/agent_recommendations.json', 'r') as f:
        data = json.load(f)
    
    # Initialize governance
    governance = PricingGovernance()
    
    print("=" * 80)
    print("VAST.AI PRICING GOVERNANCE VALIDATION")
    print("=" * 80)
    print()
    
    validated_count = 0
    rejected_count = 0
    high_risk_count = 0
    
    for rec in data['recommendations']:
        # Validate through governance
        passed, guardrails_passed, guardrails_failed, risk_level = governance.validate_decision(rec)
        
        # Create audit entry
        status = DecisionStatus.VALIDATED if passed else DecisionStatus.REJECTED
        entry = governance.create_audit_entry(
            rec, 
            guardrails_passed, 
            guardrails_failed, 
            risk_level,
            status,
            validator="ML+Governance",
            validation_notes=f"Automated validation at {datetime.utcnow().isoformat()}"
        )
        
        if passed:
            validated_count += 1
            # Auto-approve low-risk decisions
            if risk_level == RiskLevel.LOW and rec.get('confidence', 0) >= governance.config.min_confidence_auto_apply:
                governance.approve_decision(entry.decision_id, "AutoApproval")
        else:
            rejected_count += 1
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            high_risk_count += 1
            print(f"⚠️  HIGH RISK: {rec['gpu_type']} - {rec['action']} - Risk: {risk_level.value}")
            if guardrails_failed:
                print(f"   Failed: {', '.join(guardrails_failed)}")
            print()
    
    # Save audit trail
    governance._save_audit_trail()
    
    # Generate report
    report = governance.get_governance_report()
    
    print("\n" + "=" * 80)
    print("GOVERNANCE SUMMARY")
    print("=" * 80)
    print(f"Total Decisions Processed: {report['total_decisions']}")
    print(f"✅ Validated: {validated_count}")
    print(f"❌ Rejected: {rejected_count}")
    print(f"⚠️  High/Critical Risk: {high_risk_count}")
    print(f"📊 Average Confidence: {report['avg_confidence']:.1%}")
    print()
    
    print("Status Breakdown:")
    for status, count in report['status_breakdown'].items():
        print(f"  {status}: {count}")
    print()
    
    print("Risk Breakdown:")
    for risk, count in report['risk_breakdown'].items():
        print(f"  {risk}: {count}")
    print()
    
    if report.get('guardrail_violations'):
        print("Guardrail Violations:")
        for violation, count in sorted(report['guardrail_violations'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {violation}: {count}")
    
    print()
    print(f"✅ Audit trail saved to: {governance.config.audit_log_path}")
    print(f"✅ Decision history saved to: {governance.config.decision_history_path}")
    
    # Save governance report
    report_path = Path('code/vastai-pricing/data/governance_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Governance report saved to: {report_path}")


if __name__ == "__main__":
    main()
