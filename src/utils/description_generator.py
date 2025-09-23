#!/usr/bin/env python3
"""
PPE Description Generator
Generates detailed descriptions from PPE detection results
"""

import json
from typing import List, Dict, Any
from collections import Counter

class PPEDescriptionGenerator:
    def __init__(self):
        """Initialize the description generator with OSHA safety guidelines"""
        
        self.ppe_descriptions = {
            'hard_hat': 'hard hat/helmet',
            'safety_vest': 'high-visibility safety vest',
            'safety_gloves': 'protective gloves',
            'safety_boots': 'safety footwear',
            'eye_protection': 'safety glasses/goggles'
        }
        
        self.violation_descriptions = {
            'no_hard_hat': 'missing hard hat (critical head injury risk)',
            'no_safety_vest': 'missing safety vest (visibility hazard)',
            'no_safety_gloves': 'missing protective gloves (hand injury risk)',
            'no_safety_boots': 'missing safety footwear (foot injury risk)',
            'no_eye_protection': 'missing eye protection (vision injury risk)'
        }
        
        self.risk_levels = {
            'no_hard_hat': 'CRITICAL',
            'no_safety_vest': 'HIGH', 
            'no_safety_gloves': 'MEDIUM',
            'no_safety_boots': 'MEDIUM',
            'no_eye_protection': 'MEDIUM'
        }
    
    def generate_description(self, detections: List[Dict[str, Any]], 
                           include_coordinates: bool = False,
                           include_confidence: bool = False) -> Dict[str, str]:
        """
        Generate comprehensive description from detection results
        
        Args:
            detections: List of detection results with class, bbox, confidence
            include_coordinates: Whether to include bounding box info
            include_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with different description formats
        """
        
        # Analyze detections
        analysis = self._analyze_detections(detections)
        
        # Generate different description formats
        descriptions = {
            'brief': self._generate_brief_description(analysis),
            'detailed': self._generate_detailed_description(analysis, include_coordinates, include_confidence),
            'compliance': self._generate_compliance_report(analysis),
            'risk_assessment': self._generate_risk_assessment(analysis)
        }
        
        return descriptions
    
    def _analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze detection results and extract key information"""
        
        analysis = {
            'people_count': 0,
            'ppe_present': [],
            'violations': [],
            'all_detections': [],
            'confidence_scores': {}
        }
        
        for detection in detections:
            class_name = detection.get('class', detection.get('label', ''))
            confidence = detection.get('confidence', detection.get('score', 1.0))
            bbox = detection.get('bbox', detection.get('bounding_box', []))
            
            analysis['all_detections'].append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
            
            if class_name == 'person':
                analysis['people_count'] += 1
            elif class_name.startswith('no_'):
                analysis['violations'].append(class_name)
            elif class_name in self.ppe_descriptions:
                analysis['ppe_present'].append(class_name)
        
        # Count occurrences
        analysis['ppe_counts'] = Counter(analysis['ppe_present'])
        analysis['violation_counts'] = Counter(analysis['violations'])
        
        return analysis
    
    def _generate_brief_description(self, analysis: Dict[str, Any]) -> str:
        """Generate a brief, human-readable description"""
        
        people_count = analysis['people_count']
        violations = analysis['violations']
        ppe_present = analysis['ppe_present']
        
        if people_count == 0:
            return "No workers detected in the construction scene."
        
        # Start with worker count
        if people_count == 1:
            description = "Construction worker detected. "
        else:
            description = f"{people_count} construction workers detected. "
        
        # Add PPE information
        if ppe_present:
            unique_ppe = list(set(ppe_present))
            ppe_names = [self.ppe_descriptions[ppe] for ppe in unique_ppe]
            description += f"PPE observed: {', '.join(ppe_names)}. "
        
        # Add violation information
        if violations:
            description += "SAFETY VIOLATIONS DETECTED. "
        else:
            description += "No safety violations observed."
        
        return description
    
    def _generate_detailed_description(self, analysis: Dict[str, Any], 
                                     include_coordinates: bool = False,
                                     include_confidence: bool = False) -> str:
        """Generate a detailed technical description"""
        
        people_count = analysis['people_count']
        ppe_counts = analysis['ppe_counts']
        violation_counts = analysis['violation_counts']
        
        description = f"PPE Detection Analysis:\n"
        description += f"• Workers identified: {people_count}\n"
        
        if ppe_counts:
            description += f"• Safety equipment detected:\n"
            for ppe, count in ppe_counts.items():
                ppe_name = self.ppe_descriptions[ppe]
                description += f"  - {ppe_name}: {count} instance(s)\n"
        
        if violation_counts:
            description += f"• SAFETY VIOLATIONS:\n"
            for violation, count in violation_counts.items():
                violation_desc = self.violation_descriptions[violation]
                risk = self.risk_levels[violation]
                description += f"  - {violation_desc}: {count} instance(s) [Risk: {risk}]\n"
        else:
            description += f"• No safety violations detected\n"
        
        # Add technical details if requested
        if include_coordinates or include_confidence:
            description += f"\nTechnical Details:\n"
            for detection in analysis['all_detections']:
                line = f"  - {detection['class']}"
                if include_confidence:
                    line += f" (confidence: {detection['confidence']:.2f})"
                if include_coordinates:
                    bbox = detection['bbox']
                    line += f" at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                description += line + "\n"
        
        return description
    
    def _generate_compliance_report(self, analysis: Dict[str, Any]) -> str:
        """Generate OSHA compliance-focused report"""
        
        people_count = analysis['people_count']
        violations = analysis['violations']
        
        if people_count == 0:
            return "COMPLIANCE STATUS: No workers present - assessment not applicable."
        
        if not violations:
            return f"COMPLIANCE STATUS: ✅ COMPLIANT - All {people_count} worker(s) properly equipped with required PPE."
        
        # Categorize violations by risk level
        critical_violations = [v for v in violations if self.risk_levels[v] == 'CRITICAL']
        high_violations = [v for v in violations if self.risk_levels[v] == 'HIGH']
        medium_violations = [v for v in violations if self.risk_levels[v] == 'MEDIUM']
        
        report = f"COMPLIANCE STATUS: ❌ NON-COMPLIANT\n"
        report += f"Workers at risk: {people_count}\n"
        
        if critical_violations:
            report += f"CRITICAL violations: {len(critical_violations)} (immediate action required)\n"
        if high_violations:
            report += f"HIGH risk violations: {len(high_violations)}\n"
        if medium_violations:
            report += f"MEDIUM risk violations: {len(medium_violations)}\n"
        
        report += "\nRecommendations: Stop work until all PPE violations are corrected."
        
        return report
    
    def _generate_risk_assessment(self, analysis: Dict[str, Any]) -> str:
        """Generate risk assessment based on violations"""
        
        violations = analysis['violations']
        people_count = analysis['people_count']
        
        if not violations:
            return "RISK LEVEL: LOW - All required PPE properly worn."
        
        # Calculate overall risk score
        risk_scores = {'CRITICAL': 10, 'HIGH': 7, 'MEDIUM': 4}
        total_risk = sum(risk_scores[self.risk_levels[v]] for v in violations)
        
        if total_risk >= 10:
            overall_risk = "CRITICAL"
        elif total_risk >= 7:
            overall_risk = "HIGH"
        else:
            overall_risk = "MEDIUM"
        
        assessment = f"RISK LEVEL: {overall_risk}\n"
        assessment += f"Risk factors identified: {len(set(violations))}\n"
        assessment += f"Workers affected: {people_count}\n"
        
        # List specific risks
        assessment += "\nSpecific hazards:\n"
        for violation in set(violations):
            hazard = self.violation_descriptions[violation]
            risk = self.risk_levels[violation]
            assessment += f"• {hazard} [{risk} risk]\n"
        
        return assessment

def main():
    """Example usage of the PPE Description Generator"""
    
    # Example detection results (would come from your model)
    example_detections = [
        {'class': 'person', 'confidence': 0.95, 'bbox': [100, 50, 200, 400]},
        {'class': 'safety_vest', 'confidence': 0.87, 'bbox': [120, 100, 180, 250]},
        {'class': 'no_hard_hat', 'confidence': 0.92, 'bbox': [130, 50, 170, 90]},
        {'class': 'person', 'confidence': 0.89, 'bbox': [300, 80, 400, 420]},
        {'class': 'hard_hat', 'confidence': 0.84, 'bbox': [330, 80, 370, 110]},
        {'class': 'safety_gloves', 'confidence': 0.78, 'bbox': [350, 200, 380, 230]}
    ]
    
    # Generate descriptions
    generator = PPEDescriptionGenerator()
    descriptions = generator.generate_description(
        example_detections, 
        include_coordinates=True, 
        include_confidence=True
    )
    
    # Print different formats
    print("=" * 60)
    print("PPE DETECTION DESCRIPTION GENERATOR")
    print("=" * 60)
    
    print("\n📝 BRIEF DESCRIPTION:")
    print(descriptions['brief'])
    
    print("\n📋 DETAILED ANALYSIS:")
    print(descriptions['detailed'])
    
    print("\n⚖️ COMPLIANCE REPORT:")
    print(descriptions['compliance'])
    
    print("\n⚠️ RISK ASSESSMENT:")
    print(descriptions['risk_assessment'])

if __name__ == "__main__":
    main()