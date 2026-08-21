import random
import string
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionReportIssue(Action):
    def name(self) -> Text:
        return "action_report_issue"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"
        issue_type = tracker.get_slot("issue_type") or "unknown"
        issue_location = tracker.get_slot("issue_location") or "unknown"
        language = tracker.get_slot("user_language") or "en"

        # Generate a ticket ID
        ticket_id = "TK-" + "".join(random.choices(string.digits, k=8))

        issue_labels = {
            "no_signal": ("No signal", "Pas de signal"),
            "slow_internet": ("Slow internet", "Internet lent"),
            "call_drops": ("Call drops", "Appels qui coupent"),
            "wifi_issue": ("Wi-Fi not working", "Wi-Fi en panne"),
        }

        label = issue_labels.get(issue_type, (issue_type, issue_type))

        if language == "fr":
            message = (
                f"✅ **Signalement enregistré :**\n"
                f"- Référence : {ticket_id}\n"
                f"- Type : {label[1]}\n"
                f"- Localisation : {issue_location}\n"
                f"- Ligne : {phone_number}\n\n"
                f"Nos équipes techniques vont investiguer. "
                f"Vous pouvez suivre votre ticket via votre espace client ou en appelant le 3901."
            )
        else:
            message = (
                f"✅ **Issue reported:**\n"
                f"- Reference: {ticket_id}\n"
                f"- Type: {label[0]}\n"
                f"- Location: {issue_location}\n"
                f"- Line: {phone_number}\n\n"
                f"Our technical team will investigate. "
                f"You can track your ticket through your account or by calling 3901."
            )

        dispatcher.utter_message(text=message)
        return []
