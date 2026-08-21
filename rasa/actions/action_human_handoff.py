from typing import Any, Dict, List, Text

import os
from litellm import acompletion
from rasa_sdk import Action, Tracker
from rasa_sdk.events import ConversationPaused
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionHumanHandoff(Action):
    def name(self) -> Text:
        return "action_human_handoff"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        convo: List[str] = []
        for event in tracker.events:
            if event.get("event") == "user":
                user_text = str(event.get("text") or "")
                convo.append(f"user - {user_text}")
            elif event.get("event") == "bot":
                bot_text = str(event.get("text") or "")
                convo.append(f"bot - {bot_text}")
        prompt = (
            f"The following is a conversation between a bot and a human user. "
            f"Please summarise so that a human agent can easily understand the "
            f"important context. Conversation: "
            f"{convo}"
        )
        try:
            response = await acompletion(
                model="groq/openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
            )
            summarised_conversation = (
                response.choices[0].message.content or "Résumé non disponible."
            )
        except Exception as e:
            print(f"Error summarising conversation: {e}")
            summarised_conversation = "Résumé non disponible en raison d'une erreur technique."
        dispatcher.utter_message(
            response="utter_transfer_to_manager", summary=summarised_conversation
        )
        return [ConversationPaused()]
