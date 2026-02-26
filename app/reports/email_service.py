"""
Email Service
Sends reports and notifications via email
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional
from app.config import config
from app.reports.pdf_generator import PDFGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailService:
    """
    Email service for sending reports
    """

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
    ):
        """
        Initialize email service
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")

    def send_session_report(self, session, recipient_email: str) -> bool:
        """
        Send session report via email
        
        Args:
            session: Session object
            recipient_email: Recipient email address
            
        Returns:
            True if successful
        """
        try:
            # Generate PDF
            pdf_gen = PDFGenerator()
            pdf_path = pdf_gen.generate_session_report(session)

            if not pdf_path or not os.path.exists(pdf_path):
                logger.error("Failed to generate PDF for email")
                return False

            # Create email
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = f"Medical Consultation Report - Session {session.id}"

            # Email body
            body = f"""
Dear Patient,

Your medical consultation report is attached.

Session Details:
- Session ID: {session.id}
- Date: {session.created_at}
- Predicted Condition: {session.predicted_condition}
- Confidence: {session.condition_confidence*100:.1f}% if session.condition_confidence else "N/A"}

Please review the attached PDF for detailed analysis.

Disclaimer: This report is for informational purposes only and does not replace professional medical advice.

Best regards,
Vision Agentic AI System
"""

            message.attach(MIMEText(body, "plain"))

            # Attach PDF
            try:
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(pdf_path)}",
                )
                message.attach(part)
            except Exception as e:
                logger.error(f"Error attaching PDF: {e}")
                return False

            # Send email
            if not self.sender_email or not self.sender_password:
                logger.warning("Email credentials not configured")
                return False

            try:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
                server.quit()

                logger.info(f"Report sent to {recipient_email}")
                return True

            except smtplib.SMTPException as e:
                logger.error(f"SMTP error: {e}")
                return False

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_notification(self, recipient_email: str, subject: str, body: str) -> bool:
        """
        Send generic notification email
        
        Args:
            recipient_email: Recipient email
            subject: Email subject
            body: Email body
            
        Returns:
            True if successful
        """
        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            if not self.sender_email or not self.sender_password:
                logger.warning("Email credentials not configured")
                return False

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(message)
            server.quit()

            logger.info(f"Notification sent to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
