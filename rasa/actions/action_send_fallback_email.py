import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionSendFallbackEmail(Action):
    def name(self) -> Text:
        return "action_send_fallback_email"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        language = tracker.get_slot("user_language") or "en"
        phone_number = tracker.get_slot("phone_number") or "Non renseigné"
        sender_id = tracker.sender_id or "unknown"

        # Collect conversation history
        conversation_lines: list[str] = []
        for event in tracker.events:
            if event.get("event") == "user":
                user_text = str(event.get("text") or "")
                conversation_lines.append(f"👤 Client : {user_text}")
            elif event.get("event") == "bot":
                bot_text = str(event.get("text") or "")
                conversation_lines.append(f"🤖 Bot : {bot_text}")

        conversation_history = "\n".join(conversation_lines)
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")

        # Build email
        subject = f"⚠️ Message incompris — Client {phone_number} — {timestamp}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #FF6B00, #FF8C00); padding: 20px; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">⚠️ Alerte — Message incompris</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
                    Assistant Télécom — Intervention humaine requise
                </p>
            </div>

            <div style="background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0;">
                <h2 style="color: #333;">📋 Informations client</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold; width: 200px;">📞 Numéro de téléphone</td>
                        <td style="padding: 8px;">{phone_number}</td>
                    </tr>
                    <tr style="background: #f0f0f0;">
                        <td style="padding: 8px; font-weight: bold;">🆔 ID de session</td>
                        <td style="padding: 8px;">{sender_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">🕐 Date et heure</td>
                        <td style="padding: 8px;">{timestamp}</td>
                    </tr>
                    <tr style="background: #f0f0f0;">
                        <td style="padding: 8px; font-weight: bold;">🌍 Langue</td>
                        <td style="padding: 8px;">{"Français" if language == "fr" else "Anglais"}</td>
                    </tr>
                </table>
            </div>

            <div style="background: white; padding: 20px; border: 1px solid #e0e0e0; border-top: none;">
                <h2 style="color: #333;">💬 Historique complet de la conversation</h2>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; line-height: 1.8; white-space: pre-wrap;">
{conversation_history}
                </div>
            </div>

            <div style="background: #FFF3E0; padding: 15px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 10px 10px;">
                <p style="margin: 0; color: #E65100; font-weight: bold;">
                    ⚡ Action requise : Veuillez contacter le client ou traiter sa demande manuellement.
                </p>
            </div>
        </body>
        </html>
        """

        # Send email
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        alert_email_to = os.getenv("ALERT_EMAIL_TO", "")

        if not smtp_user or not smtp_password or not alert_email_to:
            if language == "fr":
                dispatcher.utter_message(
                    text="Je n'ai pas pu envoyer l'alerte email — la configuration SMTP est manquante. "
                    "Veuillez appeler le **3900** pour parler à un conseiller."
                )
            else:
                dispatcher.utter_message(
                    text="I couldn't send the email alert — SMTP configuration is missing. "
                    "Please call **3900** to speak with an advisor."
                )
            return []

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = alert_email_to
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, alert_email_to, msg.as_string())

            if language == "fr":
                dispatcher.utter_message(
                    text="📧 J'ai transmis votre conversation à notre équipe support. "
                    "Un conseiller vous recontactera dans les plus brefs délais. "
                    "Vous pouvez aussi appeler le **3900** pour une assistance immédiate."
                )
            else:
                dispatcher.utter_message(
                    text="📧 I've forwarded your conversation to our support team. "
                    "An advisor will get back to you shortly. "
                    "You can also call **3900** for immediate assistance."
                )
        except Exception as e:
            if language == "fr":
                dispatcher.utter_message(
                    text=f"❌ Erreur lors de l'envoi de l'email : {str(e)}. "
                    "Veuillez appeler le **3900** pour parler à un conseiller."
                )
            else:
                dispatcher.utter_message(
                    text=f"❌ Error sending email: {str(e)}. "
                    "Please call **3900** to speak with an advisor."
                )

        return []
