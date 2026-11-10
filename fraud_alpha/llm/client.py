"""Minimal abstract client; legacy credential discovery and alternate backends are excluded."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class Message:
    role: str
    content: str

class ChatClient:

    def __init__(self, system_prompt: Optional[str]=None):
        self._system_prompt = system_prompt
        self._dialog: List[Message] = []
        if system_prompt is not None:
            self._dialog.append(Message(role='system', content=system_prompt))

    def chat(self, content: str) -> str:
        raise NotImplementedError

    def reset(self, system_prompt: Optional[str]=None) -> None:
        if system_prompt is not None:
            self._system_prompt = system_prompt
        self._dialog = []
        if self._system_prompt is not None:
            self._dialog.append(Message(role='system', content=self._system_prompt))

    def set_system_prompt(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self.reset()

    def get_history(self) -> List[Message]:
        return self._dialog.copy()

    def set_temperature(self, temperature: float) -> None:
        pass

    def get_replay_metadata(self) -> Optional[Dict[str, Any]]:
        return None
