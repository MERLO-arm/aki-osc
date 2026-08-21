from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionChangePlan(Action):
    def name(self) -> Text:
        return "action_change_plan"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"
        desired_plan = tracker.get_slot("desired_plan") or "N/A"
        language = tracker.get_slot("user_language") or "en"

        if language == "fr":
            message = (
                f"✅ **Changement de forfait confirmé :**\n"
                f"- Ligne : {phone_number}\n"
                f"- Nouveau forfait : {desired_plan}\n"
                f"- Prise d'effet : immédiate\n\n"
                f"Votre nouveau forfait sera reflété sur votre prochaine facture. "
                f"Vous recevrez un SMS de confirmation dans quelques minutes."
            )
        else:
            message = (
                f"✅ **Plan change confirmed:**\n"
                f"- Line: {phone_number}\n"
                f"- New plan: {desired_plan}\n"
                f"- Effective: immediately\n\n"
                f"Your new plan will be reflected on your next bill. "
                f"You'll receive a confirmation SMS shortly."
            )

        dispatcher.utter_message(text=message)
        return []
