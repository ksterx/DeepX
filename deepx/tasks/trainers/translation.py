from .langmodel import LangModelTrainer


class TranslationTrainer(LangModelTrainer):
    TASK_TYPE = "translation"
