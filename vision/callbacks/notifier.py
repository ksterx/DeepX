import lightning as L


class Notifier(L.Callback):
    def __init__(self, slack_url):
        self.slack_url = slack_url

    def on_epoch_end(self, epoch, logs):
        L.utils.slack_notify(self.slack_url, f"Epoch {epoch} ended")
