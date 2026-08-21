from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionSimProblem(Action):
    def name(self) -> Text:
        return "action_sim_problem"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"
        sim_issue_type = tracker.get_slot("sim_issue_type") or "unknown"
        language = tracker.get_slot("user_language") or "en"

        responses = {
            "blocked": {
                "en": (
                    "🔒 **SIM blocked — Here's how to unblock it:**\n"
                    "1. Find your PUK code in your online account under 'My Line > PUK Code'\n"
                    "2. Or call 3900 to get it\n"
                    "3. Enter the PUK code when prompted on your phone\n"
                    "4. Set a new PIN code\n\n"
                    "⚠️ After 10 wrong PUK attempts, the SIM is permanently blocked "
                    "and you'll need a replacement."
                ),
                "fr": (
                    "🔒 **SIM bloquée — Voici comment la débloquer :**\n"
                    "1. Trouvez votre code PUK dans votre espace client > Ma ligne > Code PUK\n"
                    "2. Ou appelez le 3900 pour l'obtenir\n"
                    "3. Entrez le code PUK lorsque votre téléphone le demande\n"
                    "4. Définissez un nouveau code PIN\n\n"
                    "⚠️ Après 10 erreurs PUK, la SIM est définitivement bloquée "
                    "et un remplacement sera nécessaire."
                ),
            },
            "lost": {
                "en": (
                    "📵 **SIM lost or stolen — Immediate steps:**\n"
                    f"1. Your line ({phone_number}) will be blocked to prevent fraud\n"
                    "2. Order a replacement SIM:\n"
                    "   - Online: 10€ (delivery in 3-5 business days)\n"
                    "   - In store: free with insurance, 10€ otherwise (same-day activation)\n"
                    "3. Your phone number will be kept\n"
                    "4. If stolen, file a police report\n\n"
                    "Would you like me to connect you to an agent to block your line immediately?"
                ),
                "fr": (
                    "📵 **SIM perdue ou volée — Mesures immédiates :**\n"
                    f"1. Votre ligne ({phone_number}) sera bloquée pour éviter toute fraude\n"
                    "2. Commandez une nouvelle SIM :\n"
                    "   - En ligne : 10€ (livraison sous 3-5 jours ouvrés)\n"
                    "   - En magasin : gratuit avec assurance, 10€ sinon (activation le jour même)\n"
                    "3. Votre numéro de téléphone sera conservé\n"
                    "4. En cas de vol, déposez une plainte\n\n"
                    "Souhaitez-vous que je vous mette en relation avec un agent pour bloquer votre ligne immédiatement ?"
                ),
            },
            "format_change": {
                "en": (
                    "🔄 **SIM format change:**\n"
                    "To get a SIM in a different format (nano, micro, standard):\n"
                    "1. Visit any of our stores with your ID\n"
                    "2. Or order online (10€, delivery in 3-5 days)\n"
                    "3. Your number and data will be preserved\n"
                    "4. The old SIM is automatically deactivated"
                ),
                "fr": (
                    "🔄 **Changement de format SIM :**\n"
                    "Pour obtenir une SIM dans un autre format (nano, micro, standard) :\n"
                    "1. Rendez-vous dans n'importe quel magasin avec votre pièce d'identité\n"
                    "2. Ou commandez en ligne (10€, livraison sous 3-5 jours)\n"
                    "3. Votre numéro et vos données seront conservés\n"
                    "4. L'ancienne SIM sera automatiquement désactivée"
                ),
            },
            "esim": {
                "en": (
                    "📱 **eSIM activation:**\n"
                    "1. Check that your phone supports eSIM\n"
                    "2. Order your eSIM online or in store (10€)\n"
                    "3. You'll receive a QR code by email\n"
                    "4. Go to Settings > Mobile Network > Add Plan > Scan QR Code\n"
                    "5. Activation is instant!\n\n"
                    "✨ Bonus: You can have 2 numbers on the same phone with eSIM + physical SIM."
                ),
                "fr": (
                    "📱 **Activation eSIM :**\n"
                    "1. Vérifiez que votre téléphone supporte l'eSIM\n"
                    "2. Commandez votre eSIM en ligne ou en magasin (10€)\n"
                    "3. Vous recevrez un QR code par email\n"
                    "4. Allez dans Paramètres > Réseau mobile > Ajouter un forfait > Scanner le QR code\n"
                    "5. L'activation est instantanée !\n\n"
                    "✨ Bonus : Vous pouvez avoir 2 numéros sur le même téléphone avec eSIM + SIM physique."
                ),
            },
        }

        lang = "fr" if language == "fr" else "en"
        response = responses.get(sim_issue_type, {})

        if response:
            message = response.get(lang, response.get("en", ""))
        else:
            if lang == "fr":
                message = (
                    f"Je comprends que vous avez un problème de SIM ({sim_issue_type}). "
                    "Je vous recommande d'appeler le 3900 ou de vous rendre en magasin "
                    "pour une assistance immédiate."
                )
            else:
                message = (
                    f"I understand you have a SIM issue ({sim_issue_type}). "
                    "I recommend calling 3900 or visiting a store for immediate assistance."
                )

        dispatcher.utter_message(text=message)
        return []
