"""
PDF Report Generator
Creates formatted PDF reports from session data
"""

import os
from datetime import datetime
from typing import Optional
from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. Install with: pip install reportlab")


class PDFGenerator:
    """
    Generate PDF reports from session data
    """

    def __init__(self):
        """Initialize PDF generator"""
        os.makedirs(config.REPORTS_PATH, exist_ok=True)

    def generate_session_report(self, session) -> Optional[str]:
        """
        Generate PDF report for a session
        
        Args:
            session: Session object from database
            
        Returns:
            Path to generated PDF or None
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available")
            return None

        try:
            filename = f"report_session_{session.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(config.REPORTS_PATH, filename)

            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=30,
                alignment=1,
            )
            elements.append(
                Paragraph("Medical Consultation Report", title_style)
            )
            elements.append(Spacer(1, 0.2 * inch))

            # Session Information
            elements.append(Paragraph("<b>Session Information</b>", styles["Heading2"]))
            session_info = [
                ["Session ID:", str(session.id)],
                ["Date:", session.created_at.strftime("%Y-%m-%d %H:%M")],
                ["Patient ID:", str(session.patient_id)],
            ]
            session_table = Table(session_info, colWidths=[2 * inch, 4 * inch])
            session_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(session_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Chief Complaint & Symptoms
            elements.append(Paragraph("<b>Symptoms & Complaints</b>", styles["Heading2"]))
            symptoms_text = f"<b>Chief Complaint:</b> {session.chief_complaint}<br/>"
            if session.symptoms:
                symptoms_text += f"<b>Symptoms:</b> {', '.join(session.symptoms)}<br/>"
            symptoms_text += f"<b>Duration:</b> {session.symptom_duration}<br/>"
            symptoms_text += f"<b>Severity:</b> {session.severity}"
            elements.append(Paragraph(symptoms_text, styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

            # Analysis Results
            elements.append(Paragraph("<b>Analysis Results</b>", styles["Heading2"]))

            results = [
                ["Metric", "Result"],
                ["Predicted Condition", session.predicted_condition or "N/A"],
                ["Confidence", f"{session.condition_confidence*100:.1f}%" if session.condition_confidence else "N/A"],
                ["Patient Emotion", session.emotion or "N/A"],
                ["Speech Sentiment", session.sentiment or "N/A"],
                ["Eye Strain Level", str(session.eye_strain_score) if session.eye_strain_score else "N/A"],
            ]

            results_table = Table(results, colWidths=[3 * inch, 3 * inch])
            results_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]
                )
            )
            elements.append(results_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Recommendation
            elements.append(
                Paragraph("<b>Clinical Recommendation</b>", styles["Heading2"])
            )
            recommendation_text = f"<b>Suggested Remedy:</b> {session.suggested_medication or 'N/A'}<br/>"
            recommendation_text += f"<b>Dosage:</b> {session.medication_dosage or 'N/A'}<br/>"
            if session.medication_warnings:
                recommendation_text += f"<b>Warnings:</b> {session.medication_warnings}"
            elements.append(Paragraph(recommendation_text, styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

            # Safety Status
            elements.append(Paragraph("<b>Safety Status</b>", styles["Heading2"]))
            safety_status = "✓ PASSED" if session.safety_check_passed else "✗ REVIEW REQUIRED"
            elements.append(Paragraph(f"Safety Check: {safety_status}", styles["Normal"]))
            if session.safety_warnings:
                elements.append(Paragraph(f"Warnings: {session.safety_warnings}", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

            # Disclaimer
            elements.append(Spacer(1, 0.3 * inch))
            disclaimer_style = ParagraphStyle(
                "Disclaimer",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
                alignment=0,
            )
            elements.append(
                Paragraph(
                    "<i>This report is generated by an AI medical analysis system for homeopathic consultation. "
                    "It is not a substitute for professional medical advice. Please consult a licensed healthcare provider.</i>",
                    disclaimer_style,
                )
            )

            # Build PDF
            doc.build(elements)
            logger.info(f"Generated report: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return None
