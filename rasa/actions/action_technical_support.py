from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionTechnicalSupport(Action):
    def name(self) -> Text:
        return "action_technical_support"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        phone_number = tracker.get_slot("phone_number") or "N/A"
        tech_issue = tracker.get_slot("tech_issue_description") or ""
        language = tracker.get_slot("user_language") or "en"

        # Basic keyword-based troubleshooting
        tech_issue_lower = tech_issue.lower()

        if any(w in tech_issue_lower for w in ["wifi", "wi-fi", "box", "internet", "fibre"]):
            if language == "fr":
                message = (
                    "🔧 **Dépannage Internet / Wi-Fi :**\n\n"
                    "Essayez ces étapes :\n"
                    "1. **Redémarrez** votre box en la débranchant 30 secondes\n"
                    "2. Vérifiez les **voyants** : le voyant Internet doit être vert fixe\n"
                    "3. Rapprochez-vous de la box ou utilisez le **Wi-Fi 5 GHz**\n"
                    "4. Vérifiez les **câbles** (Ethernet, fibre)\n"
                    "5. Réduisez le nombre d'appareils connectés\n\n"
                    "Si le problème persiste, appelez le **3901** pour un diagnostic à distance."
                )
            else:
                message = (
                    "🔧 **Internet / Wi-Fi Troubleshooting:**\n\n"
                    "Try these steps:\n"
                    "1. **Restart** your box by unplugging it for 30 seconds\n"
                    "2. Check the **indicator lights**: Internet light should be solid green\n"
                    "3. Move closer to the box or switch to **5 GHz Wi-Fi**\n"
                    "4. Check all **cables** (Ethernet, fiber)\n"
                    "5. Reduce the number of connected devices\n\n"
                    "If the issue persists, call **3901** for a remote diagnostic."
                )
        elif any(w in tech_issue_lower for w in ["appel", "call", "voix", "voice", "micro"]):
            if language == "fr":
                message = (
                    "🔧 **Dépannage Appels :**\n\n"
                    "1. Vérifiez que le **mode avion** est désactivé\n"
                    "2. Redémarrez votre téléphone\n"
                    "3. Vérifiez que vous n'avez pas activé le **renvoi d'appels**\n"
                    "4. Testez avec un autre téléphone si possible\n"
                    "5. Vérifiez la **couverture réseau** dans votre zone\n\n"
                    "Si le problème persiste, contactez le **3901**."
                )
            else:
                message = (
                    "🔧 **Call Troubleshooting:**\n\n"
                    "1. Make sure **airplane mode** is off\n"
                    "2. Restart your phone\n"
                    "3. Check that **call forwarding** is not enabled\n"
                    "4. Test with another phone if possible\n"
                    "5. Check **network coverage** in your area\n\n"
                    "If the issue persists, contact **3901**."
                )
        elif any(w in tech_issue_lower for w in ["tv", "décodeur", "decoder", "chaine", "channel"]):
            if language == "fr":
                message = (
                    "🔧 **Dépannage TV :**\n\n"
                    "1. Vérifiez que le décodeur est bien **branché et allumé**\n"
                    "2. Vérifiez la connexion **HDMI** avec votre TV\n"
                    "3. Sélectionnez la bonne **source HDMI** sur votre TV\n"
                    "4. Redémarrez le décodeur\n"
                    "5. Vérifiez votre débit internet (minimum **10 Mbit/s** pour la TV)\n\n"
                    "Si le problème persiste, contactez le **3901**."
                )
            else:
                message = (
                    "🔧 **TV Troubleshooting:**\n\n"
                    "1. Check that the decoder is **plugged in and powered on**\n"
                    "2. Check the **HDMI** connection to your TV\n"
                    "3. Select the correct **HDMI source** on your TV\n"
                    "4. Restart the decoder\n"
                    "5. Check your internet speed (minimum **10 Mbit/s** for TV)\n\n"
                    "If the issue persists, contact **3901**."
                )
        else:
            if language == "fr":
                message = (
                    f"🔧 **Support technique :**\n\n"
                    f"J'ai bien noté votre problème : « {tech_issue} »\n\n"
                    f"Voici les étapes générales :\n"
                    f"1. **Redémarrez** votre appareil\n"
                    f"2. Vérifiez les **mises à jour** disponibles\n"
                    f"3. Réinitialisez les **paramètres réseau**\n\n"
                    f"Pour un diagnostic approfondi, contactez le **3901** "
                    f"(du lundi au samedi, 8h-20h)."
                )
            else:
                message = (
                    f"🔧 **Technical support:**\n\n"
                    f"I've noted your issue: \"{tech_issue}\"\n\n"
                    f"General troubleshooting steps:\n"
                    f"1. **Restart** your device\n"
                    f"2. Check for available **updates**\n"
                    f"3. Reset your **network settings**\n\n"
                    f"For an in-depth diagnostic, call **3901** "
                    f"(Monday to Saturday, 8am-8pm)."
                )

        dispatcher.utter_message(text=message)
        return []
