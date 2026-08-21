import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionCheckBill(Action):
    def name(self) -> Text:
        return "action_check_bill"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"
        language = tracker.get_slot("user_language") or "en"

        # Simulated billing data
        plan_charges = random.choice(["9,99", "19,99", "29,99"])
        extra_charges = random.choice(["0,00", "2,50", "5,80", "12,30"])
        total = f"{float(plan_charges.replace(',', '.')) + float(extra_charges.replace(',', '.')):.2f}".replace(
            ".", ","
        )

        now = datetime.now()
        billing_period = f"01/{now.month:02d}/{now.year} - {now.day:02d}/{now.month:02d}/{now.year}"
        payment_status = random.choice(["Paid", "Pending", "Overdue"])
        payment_method = random.choice(
            ["Direct debit", "Credit card", "Bank transfer"]
        )

        if language == "fr":
            status_map = {"Paid": "Payée", "Pending": "En attente", "Overdue": "En retard"}
            method_map = {
                "Direct debit": "Prélèvement automatique",
                "Credit card": "Carte bancaire",
                "Bank transfer": "Virement bancaire",
            }
            message = (
                f"💰 **Résumé de votre facture ({phone_number}) :**\n"
                f"- Période : {billing_period}\n"
                f"- Frais d'abonnement : {plan_charges}€\n"
                f"- Frais supplémentaires : {extra_charges}€\n"
                f"- **Total : {total}€**\n"
                f"- Statut : {status_map.get(payment_status, payment_status)}\n"
                f"- Mode de paiement : {method_map.get(payment_method, payment_method)}"
            )
        else:
            message = (
                f"💰 **Your bill summary ({phone_number}):**\n"
                f"- Billing period: {billing_period}\n"
                f"- Plan charges: {plan_charges}€\n"
                f"- Extra charges: {extra_charges}€\n"
                f"- **Total: {total}€**\n"
                f"- Payment status: {payment_status}\n"
                f"- Payment method: {payment_method}"
            )

        dispatcher.utter_message(text=message)
        return []
