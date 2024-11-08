import heapq
import json
import threading
import queue

class TopSequencesTracker:
    def __init__(self, max_size=1000, filename='top_sequences.json'):
        self.max_size = max_size
        self.filename = filename
        self.sequences = []
        self.load_from_file()
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def add_sequence(self, amino_acid, value1, value2):
        # Use value1 directly for max-heap behavior
        if len(self.sequences) < self.max_size:
            heapq.heappush(self.sequences, (value1, amino_acid, value2))
        elif value1 > self.sequences[0][0]:
            heapq.heapreplace(self.sequences, (value1, amino_acid, value2))
        self.save_queue.put(self.get_top_sequences())

    def get_top_sequences(self):
        return sorted([(seq[1], seq[0], seq[2]) for seq in self.sequences],
                      key=lambda x: x[1], reverse=True)

    def _save_worker(self):
        while True:
            data = self.save_queue.get()
            with open(self.filename, 'w') as f:
                json.dump(data, f)
            self.save_queue.task_done()

    def load_from_file(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
            self.sequences = [(value1, amino_acid, value2)
                              for amino_acid, value1, value2 in data]
            heapq.heapify(self.sequences)
        except FileNotFoundError:
            print(f"File {self.filename} not found. Starting with an empty list.")