import sys
import os
import subprocess
import time
import psutil
import torch
import shutil
import whisper
import logging

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox, QStatusBar
)

# Configurazione base del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Abilita cuDNN benchmark se CUDA è disponibile
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("GPU disponibile: utilizzo di cuDNN benchmark abilitato.")
else:
    logging.info("GPU non disponibile: utilizzo della CPU.")

# Lista fissa dei modelli disponibili
ALL_MODELS = ["tiny", "base", "small", "medium", "large"]

# ---------------------------
# Funzioni di utilità
# ---------------------------
def check_recommended_model():
    """
    Controlla GPU e RAM per suggerire un modello.
    Restituisce una tupla: (modello consigliato, specifiche del sistema)
    """
    specs = {
        "cpu": psutil.cpu_count(logical=False),
        "ram": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
        "gpu": None,
        "vram": None
    }
    recommended = "tiny"
    try:
        if torch.cuda.is_available():
            specs["gpu"] = torch.cuda.get_device_name(0)
            specs["vram"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if specs["vram"] >= 8:
                recommended = "large"
            elif specs["vram"] >= 6:
                recommended = "medium"
            elif specs["vram"] >= 4:
                recommended = "small"
            else:
                recommended = "base"
        else:
            ram_avail = psutil.virtual_memory().available / (1024 ** 3)
            recommended = "small" if ram_avail >= 8 else "base"
    except Exception as e:
        logging.error("Errore in check_recommended_model: " + str(e))
        recommended = "tiny"
    logging.info(f"Modello consigliato: {recommended}")
    return recommended, specs

def convert_file_if_needed(file_path):
    """
    Converte il file in formato .wav tramite ffmpeg se si tratta di un video.
    Se il file è già in un formato audio supportato (.mp3, .wav, .m4a, .flac),
    lo restituisce così com'è.
    """
    logging.info(f"Verifica conversione file: {file_path}")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg non è stato trovato nel PATH. Assicurati che sia installato e configurato correttamente.")
    
    audio_ext = [".mp3", ".wav", ".m4a", ".flac"]
    video_ext = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(file_path)[1].lower()

    if ext in audio_ext:
        logging.info("Il file è in formato audio supportato; nessuna conversione necessaria.")
        return file_path
    elif ext in video_ext:
        new_file = os.path.splitext(file_path)[0] + ".wav"
        command = ["ffmpeg", "-y", "-i", file_path, new_file]
        try:
            logging.info(f"Avvio conversione video -> audio: {' '.join(command)}")
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"Conversione completata: {new_file}")
            return new_file
        except subprocess.CalledProcessError as e:
            logging.error("Errore nella conversione con ffmpeg: " + str(e))
            return None
    else:
        logging.warning("Formato file non supportato per la conversione.")
        return None

# ---------------------------
# Gestione del modello con caching
# ---------------------------
class ModelManager:
    def __init__(self):
        self.cache_folder = "models"
        os.makedirs(self.cache_folder, exist_ok=True)
        self.loaded_model = None
        self.current_model_name = None

    def is_model_downloaded(self, model_name):
        model_path = os.path.join(self.cache_folder, model_name + ".model")
        exists = os.path.exists(model_path)
        logging.info(f"Verifica modello '{model_name}' scaricato: {exists}")
        return exists

    def mark_model_as_downloaded(self, model_name):
        model_path = os.path.join(self.cache_folder, model_name + ".model")
        with open(model_path, "w") as f:
            f.write("downloaded")
        logging.info(f"Modello '{model_name}' marcato come scaricato.")

    def load_model(self, model_name, progress_callback=None):
        if self.loaded_model is not None and self.current_model_name == model_name:
            logging.info("Utilizzo modello già in cache.")
            return self.loaded_model
        if progress_callback:
            progress_callback(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Caricamento modello '{model_name}' sul device '{device}'...")
        model = whisper.load_model(model_name, device=device)
        self.loaded_model = model
        self.current_model_name = model_name
        if progress_callback:
            progress_callback(100)
        logging.info(f"Modello '{model_name}' caricato con successo su '{device}'.")
        return model

# Istanza globale per gestire il modello
model_manager = ModelManager()

# ---------------------------
# Thread per il download del modello
# ---------------------------
class DownloaderThread(QThread):
    progressUpdated = pyqtSignal(int)
    downloadFinished = pyqtSignal(bool, str)
    
    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self._is_interrupted = False

    def run(self):
        logging.info(f"DownloaderThread iniziato per il modello '{self.model_name}'.")
        try:
            # Durante il download, simuliamo aggiornamenti di progresso
            self.progressUpdated.emit(0)
            model = model_manager.load_model(self.model_name, progress_callback=lambda p: self.progressUpdated.emit(p))
            logging.info("Download e caricamento del modello completati.")
            self.downloadFinished.emit(True, "")
        except Exception as e:
            logging.error("DownloaderThread errore: " + str(e))
            self.downloadFinished.emit(False, str(e))

# ---------------------------
# Thread per la trascrizione con Whisper
# ---------------------------
class TranscriptionThread(QThread):
    progressUpdated = pyqtSignal(int)
    transcriptionFinished = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, model_name, file_path, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.file_path = file_path
        self._is_interrupted = False

    def run(self):
        logging.info(f"TranscriptionThread iniziato per il file '{self.file_path}' con il modello '{self.model_name}'.")
        try:
            self.progressUpdated.emit(0)
            model = model_manager.load_model(self.model_name, progress_callback=lambda p: self.progressUpdated.emit(p))
            # Una volta caricato il modello, assicuriamoci che la barra arrivi a 100%
            self.progressUpdated.emit(100)
            if self._is_interrupted:
                self.errorOccurred.emit("Trascrizione interrotta.")
                logging.warning("Trascrizione interrotta prima di iniziare.")
                return
            # Simuliamo un aggiornamento prima della trascrizione vera
            self.progressUpdated.emit(50)
            use_fp16 = torch.cuda.is_available()
            logging.info("Inizio trascrizione (FP16=" + str(use_fp16) + ")...")
            result = model.transcribe(self.file_path, fp16=use_fp16, language="it", temperature=0, beam_size=5, best_of=5)
            final_text = result.get("text", "")
            self.progressUpdated.emit(100)
            logging.info("Trascrizione completata con successo.")
            self.transcriptionFinished.emit(final_text)
        except Exception as e:
            logging.error("Errore in TranscriptionThread: " + str(e))
            self.errorOccurred.emit(str(e))

    def interrupt(self):
        self._is_interrupted = True
        logging.info("Richiesta di interruzione della trascrizione ricevuta.")

# ---------------------------
# Finestra Principale (GUI)
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trascrizione Audio/Video")
        self.file_path = None
        self.downloader_thread = None
        self.transcription_thread = None
        self.last_progress_update = time.time()
        self.initUI()
        self.setup_model_lists()
        self.start_monitor()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Pannello per le specifiche del sistema
        specs_layout = QHBoxLayout()
        self.specs_labels = {
            "cpu": QLabel("CPU: N/D"),
            "ram": QLabel("RAM: N/D"),
            "gpu": QLabel("GPU: N/D"),
            "vram": QLabel("VRAM: N/D")
        }
        for key in ["cpu", "ram", "gpu", "vram"]:
            specs_layout.addWidget(self.specs_labels[key])
        main_layout.addLayout(specs_layout)

        # Pannello superiore: modelli scaricabili e modelli scaricati
        top_layout = QHBoxLayout()
        self.combo_available = QComboBox()
        top_layout.addWidget(QLabel("Modelli da scaricare:"))
        top_layout.addWidget(self.combo_available)
        self.combo_downloaded = QComboBox()
        top_layout.addWidget(QLabel("Modelli scaricati:"))
        top_layout.addWidget(self.combo_downloaded)
        main_layout.addLayout(top_layout)

        # Etichetta per il modello consigliato
        self.label_recommended = QLabel("Modello consigliato: N/D")
        main_layout.addWidget(self.label_recommended)

        # Pannello dei pulsanti
        button_layout = QHBoxLayout()
        self.btn_download = QPushButton("Scarica modello selezionato")
        self.btn_download.clicked.connect(self.download_model)
        button_layout.addWidget(self.btn_download)
        self.btn_choose_file = QPushButton("Scegli File")
        self.btn_choose_file.clicked.connect(self.choose_file)
        button_layout.addWidget(self.btn_choose_file)
        self.btn_transcribe = QPushButton("Trascrivi")
        self.btn_transcribe.clicked.connect(self.start_transcription)
        button_layout.addWidget(self.btn_transcribe)
        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self.cancel_transcription)
        button_layout.addWidget(self.btn_cancel)
        self.btn_save = QPushButton("Salva Trascrizione")
        self.btn_save.clicked.connect(self.save_transcription)
        button_layout.addWidget(self.btn_save)
        main_layout.addLayout(button_layout)

        # Label per il percorso del file selezionato
        self.label_file_path = QLabel("Nessun file selezionato")
        main_layout.addWidget(self.label_file_path)

        # Barra di avanzamento
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Area di testo per l'output della trascrizione
        self.text_output = QTextEdit()
        main_layout.addWidget(self.text_output)

        # Status bar per messaggi
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_model_lists(self):
        recommended, specs = check_recommended_model()
        self.specs_labels["cpu"].setText(f"CPU: {psutil.cpu_count(logical=False)} core fisici")
        self.specs_labels["ram"].setText(f"RAM: {specs['ram']} GB")
        self.specs_labels["gpu"].setText(f"GPU: {specs['gpu'] or 'Non disponibile'}")
        self.specs_labels["vram"].setText(f"VRAM: {specs['vram'] or 'N/D'} GB")
        
        downloaded = [m for m in ALL_MODELS if model_manager.is_model_downloaded(m)]
        available = [m for m in ALL_MODELS if not model_manager.is_model_downloaded(m)]
        self.combo_downloaded.clear()
        self.combo_available.clear()
        self.combo_downloaded.addItems(downloaded)
        self.combo_available.addItems(available)
        self.label_recommended.setText(f"Modello consigliato: {recommended} (basato sulle specifiche del sistema)")
        if recommended in downloaded:
            index = self.combo_downloaded.findText(recommended)
            if index != -1:
                self.combo_downloaded.setCurrentIndex(index)
        logging.info(f"Modelli scaricati: {downloaded}; modelli disponibili: {available}")

    def start_monitor(self):
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.check_progress_stuck)
        self.monitor_timer.start(1000)  # Controlla ogni secondo

    def check_progress_stuck(self):
        elapsed = time.time() - self.last_progress_update
        if elapsed > 20:  # Timeout aumentato a 20 secondi
            self.status_bar.showMessage("Operazione in corso da troppo tempo. Verifica se il programma è bloccato...", 5000)
            logging.warning("Nessun aggiornamento di progresso per più di 20 secondi.")

    def download_model(self):
        if self.combo_available.count() == 0:
            QMessageBox.information(self, "Informazione", "Non ci sono modelli da scaricare.")
            return
        model_name = self.combo_available.currentText()
        self.btn_download.setEnabled(False)
        self.text_output.append(f"Inizio download del modello '{model_name}'")
        logging.info(f"Inizio download del modello '{model_name}'.")
        self.downloader_thread = DownloaderThread(model_name)
        self.downloader_thread.progressUpdated.connect(self.on_download_progress)
        self.downloader_thread.downloadFinished.connect(self.on_download_finished)
        self.downloader_thread.start()

    def on_download_progress(self, value):
        self.progress_bar.setValue(value)
        self.last_progress_update = time.time()
        self.status_bar.showMessage(f"Download in corso: {value}%", 2000)
        logging.info(f"Download progress: {value}%")

    def on_download_finished(self, success, error):
        if success:
            self.text_output.append("Download completato.")
            logging.info("Download completato con successo.")
            model_manager.mark_model_as_downloaded(self.combo_available.currentText())
            self.status_bar.showMessage("Download completato.", 3000)
        else:
            self.text_output.append(f"Download fallito: {error}")
            logging.error("Download fallito: " + error)
            self.status_bar.showMessage("Download fallito.", 3000)
        self.setup_model_lists()
        self.progress_bar.setValue(0)
        self.btn_download.setEnabled(True)

    def choose_file(self):
        options = QFileDialog.Option(0)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona file audio/video", "",
            "Audio/Video Files (*.mp3 *.wav *.m4a *.flac *.mp4 *.avi *.mov *.mkv)", options=options)
        if file_path:
            logging.info(f"File selezionato: {file_path}")
            try:
                converted_file = convert_file_if_needed(file_path)
            except FileNotFoundError as e:
                QMessageBox.critical(self, "Errore", str(e))
                logging.error(str(e))
                return
            if converted_file:
                self.file_path = converted_file
                self.label_file_path.setText(self.file_path)
                self.text_output.append("File selezionato: " + self.file_path)
                self.status_bar.showMessage("File caricato correttamente.", 3000)
                logging.info("File convertito e caricato correttamente.")
            else:
                self.text_output.append("Errore nella conversione del file.")
                QMessageBox.warning(self, "Errore", "Impossibile convertire il file nel formato richiesto.")
                logging.error("Errore nella conversione del file.")

    def start_transcription(self):
        if not self.file_path:
            QMessageBox.warning(self, "Attenzione", "Seleziona un file prima di trascrivere.")
            return
        if self.combo_downloaded.count() == 0:
            QMessageBox.warning(self, "Attenzione", "Non ci sono modelli scaricati. Scarica un modello prima.")
            return
        model_name = self.combo_downloaded.currentText()
        self.text_output.append("Inizio trascrizione...")
        logging.info(f"Inizio trascrizione con il modello '{model_name}' per il file '{self.file_path}'.")
        self.progress_bar.setValue(0)
        self.transcription_thread = TranscriptionThread(model_name, self.file_path)
        self.transcription_thread.progressUpdated.connect(self.on_transcription_progress)
        self.transcription_thread.transcriptionFinished.connect(self.on_transcription_finished)
        self.transcription_thread.errorOccurred.connect(self.on_transcription_error)
        self.transcription_thread.start()

    def cancel_transcription(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.interrupt()
            self.text_output.append("Trascrizione annullata.")
            self.status_bar.showMessage("Trascrizione annullata.", 3000)
            self.progress_bar.setValue(0)
            logging.info("Trascrizione annullata su richiesta.")

    def on_transcription_progress(self, value):
        self.progress_bar.setValue(value)
        self.last_progress_update = time.time()
        self.status_bar.showMessage(f"Trascrizione in corso: {value}%", 2000)
        logging.info(f"Trascrizione progress: {value}%")

    def on_transcription_finished(self, text):
        self.text_output.append("Trascrizione completata:")
        self.text_output.append(text)
        self.status_bar.showMessage("Trascrizione completata.", 3000)
        self.progress_bar.setValue(0)
        logging.info("Trascrizione completata con successo.")

    def on_transcription_error(self, error):
        self.text_output.append("Errore: " + error)
        self.status_bar.showMessage("Errore nella trascrizione.", 3000)
        self.progress_bar.setValue(0)
        logging.error("Errore nella trascrizione: " + error)

    def save_transcription(self):
        if not self.text_output.toPlainText():
            QMessageBox.warning(self, "Attenzione", "Non c'è alcuna trascrizione da salvare.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Salva Trascrizione", "", "File di testo (*.txt)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.text_output.toPlainText())
                self.status_bar.showMessage("Trascrizione salvata.", 3000)
                logging.info(f"Trascrizione salvata nel file: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore nel salvataggio: {e}")
                logging.error("Errore nel salvataggio della trascrizione: " + str(e))

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.resize(800, 600)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical("Errore critico: " + str(e))
