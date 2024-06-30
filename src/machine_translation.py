"""
script for machine translation
"""
import warnings
import json
warnings.filterwarnings("ignore")


class MachineTranslation:
    """
    Text2Text generation using opus-MT from Helsinki-NLP
    """
    def __init__(self, model_path: str) -> None:
        self.pretrain = json.load(open(model_path, "r", encoding="utf-8"))

    def machine_translation(self, *args) -> None:
        """
        Translate source text to target language
        """
        pass

    def get_output(self, *args) -> None:
        """
        Translate all source text to target language
        """
        pass
