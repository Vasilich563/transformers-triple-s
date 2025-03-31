from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileDeletedEvent, FileCreatedEvent, FileModifiedEvent, FileMovedEvent


class CrawlerHandler(FileSystemEventHandler):

    def __init__(self):
        self.i = 0
        # TODO rename is move
        # TODO modify = delete old + create new
        # TODO move = (delete src + delete dest) + create dest

    def on_created(self, event: FileCreatedEvent):
        if not event.is_directory:
            print(self.i, "on_created", event.src_path, event.dest_path)
            self.i += 1

    def on_deleted(self, event: FileDeletedEvent):
        if not event.is_directory:
            print(self.i, "on_deleted", event.src_path, event.dest_path)
            self.i += 1

    def on_modified(self, event: FileModifiedEvent):
        if not event.is_directory:
            print(self.i, "on_modified", event.src_path, event.dest_path)
            self.i += 1

    def on_moved(self, event: FileMovedEvent):
        if not event.is_directory:
            print(self.i, "on_moved", event.src_path, event.dest_path)
            self.i += 1


handler = CrawlerHandler()
observer = Observer()
observer.schedule(handler, path="/home/yackub/PycharmProjects/Diploma/temp", recursive=True)
observer.start()

while True:
    try:
        pass
    except Exception:
        observer.stop()



