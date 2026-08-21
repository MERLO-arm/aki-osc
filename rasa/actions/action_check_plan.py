import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.types import DomainDict


class ActionCheckPlan(Action):
    def name(self) -> Text:
        return "action_check_plan"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"

        # Simulated plan data (in production, this would query a database or API)
        plans = [
            {
                "name": "Forfait Essentiel",
                "data_used": "12.5 Go",
                "data_total": "20 Go",
                "price": "9,99€",
            },
            {
                "name": "Forfait Confort",
                "data_used": "35.2 Go",
                "data_total": "80 Go",
                "price": "19,99€",
            },
            {
                "name": "Forfait Premium",
                "data_used": "78.1 Go",
                "data_total": "150 Go",
                "price": "29,99€",
            },
        ]

        plan = random.choice(plans)
        next_billing = (datetime.now() + timedelta(days=random.randint(5, 25))).strftime(
            "%d/%m/%Y"
        )

        language = tracker.get_slot("user_language") or "en"

        if language == "fr":
            message = (
                f"📱 **Détails de votre forfait ({phone_number}) :**\n"
                f"- Forfait : {plan['name']}\n"
                f"- Data utilisée : {plan['data_used']} / {plan['data_total']}\n"
                f"- Appels : illimités\n"
                f"- SMS : illimités\n"
                f"- Montant mensuel : {plan['price']}/mois\n"
                f"- Prochaine facturation : {next_billing}"
            )
        else:
            message = (
                f"📱 **Your plan details ({phone_number}):**\n"
                f"- Plan: {plan['name']}\n"
                f"- Data used: {plan['data_used']} / {plan['data_total']}\n"
                f"- Calls: unlimited\n"
                f"- SMS: unlimited\n"
                f"- Monthly fee: {plan['price']}/month\n"
                f"- Next billing date: {next_billing}"
            )

        dispatcher.utter_message(text=message)
        return []
