from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileDeletedEvent, FileCreatedEvent, FileModifiedEvent, FileMovedEvent
from backend.embedding_system.embedding_system import EmbeddingSystem
from threading import Thread
import PyPDF2
import docx
from bs4 import BeautifulSoup


class CrawlerHandler(FileSystemEventHandler):


    def __init__(self):
        self.txt_postfix = ".txt"
        self.pdf_postfix = ".pdf"
        self.html_postfix = ".html"
        self.docx_postfix = ".docx"
        self.postfixes = [self.txt_postfix, self.pdf_postfix, self.docx_postfix, self.html_postfix]

    @staticmethod
    def _extract_text_from_txt(path_to_file):
        with open(path_to_file, 'r') as fin:
            text = fin.read()
        return text

    @staticmethod
    def _extract_text_from_pdf(path_to_file):
        with open(path_to_file, 'rb') as fin:
            pdfReader = PyPDF2.PdfReader(fin)
            text = " ".join(
                [page.extract_text() for page in pdfReader.pages]
            )
        return text

    @staticmethod
    def _extract_text_from_html(path_to_file):
        with open(path_to_file, 'r') as fin:
            soup = BeautifulSoup(fin.read(), features="html.parser")

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text

    @staticmethod
    def _extract_text_from_docx(path_to_file):
        doc = docx.Document(path_to_file)
        text = ' '.join(
            [paragraph.text for paragraph in doc.paragraphs]
        )
        return text


    def _extract_text(self, path_to_file):
        if path_to_file.endswith(self.txt_postfix):
            return self._extract_text_from_txt(path_to_file)
        elif path_to_file.endswith(self.pdf_postfix):
            return self._extract_text_from_pdf(path_to_file)
        elif path_to_file.endswith(self.html_postfix):
            return self._extract_text_from_html(path_to_file)
        elif path_to_file.endswith(self.docx_postfix):
            return self._extract_text_from_docx(path_to_file)
        else:
            return None


    def on_created(self, event: FileCreatedEvent):
        if not event.is_directory:
            text = self._extract_text(event.src_path)
            print("INSERT")
            print(text)
            if text is not None:
                delete_daemon = Thread(target=EmbeddingSystem.index_new_text, args=(text, event.src_path), daemon=True)
                delete_daemon.start()


    def on_deleted(self, event: FileDeletedEvent):
        if not event.is_directory:
            print("DELETE")
            for postfix in self.postfixes:
                print(f"DELETE {postfix}")
                if event.src_path.endswith(postfix):
                    print("DELETE")
                    delete_daemon = Thread(target=EmbeddingSystem.remove_document, args=(event.src_path,), daemon=True)
                    delete_daemon.start()
                    break



    def _handle_on_modified(self, event: FileModifiedEvent):
        # modified is remove old and index new
        remove_thread = Thread(target=EmbeddingSystem.remove_document, args=(event.src_path,))
        remove_thread.start()

        text = self._extract_text(event.src_path)

        remove_thread.join()
        if text is not None:
            EmbeddingSystem.index_new_text(text, event.src_path)


    def on_modified(self, event: FileModifiedEvent):
        # modified is remove old and index new
        if not event.is_directory:
            for postfix in self.postfixes:
                if event.src_path.endswith(postfix):
                    modified_daemon = Thread(target=self._handle_on_modified, args=(event,), daemon=True)
                    modified_daemon.start()


    def _handle_on_moved(self, event: FileModifiedEvent):
        # on moved is delete dest, delete src (if src can be stored in base) and index dest
        remove_dest_thread = Thread(target=EmbeddingSystem.remove_document, args=(event.dest_path,))
        remove_dest_thread.start()

        for postfix in self.postfixes:
            if event.src_path.endswith(postfix):
                remove_src_thread = Thread(target=EmbeddingSystem.remove_document, args=(event.src_path,))
                remove_src_thread.start()
                remove_src_thread.join()

        remove_dest_thread.join()

        text = self._extract_text(event.dest_path)
        if text is not None:
            EmbeddingSystem.index_new_text(text, event.src_path)


    def on_moved(self, event: FileMovedEvent):
        # on moved is delete dest, delete src (if src can be stored in base) and index dest
        if not event.is_directory:
            for postfix in self.postfixes:
                if event.dest_path.endswith(postfix):
                    moved_daemon = Thread(target=self._handle_on_moved, args=(event,), daemon=True)
                    moved_daemon.start()


def observe_directory(path):
    handler = CrawlerHandler()
    observer = Observer()
    observer.schedule(handler, path=path, recursive=True)
    observer.start()

    while True:
        try:
            pass
        except Exception:
            observer.stop()

def observe_directory_daemon(path):
    daemon = Thread(target=observe_directory, args=(path,), daemon=True)
    daemon.start()


if __name__ == "__main__":
    observe_directory("/home/yackub/PycharmProjects/Diploma/temp")



