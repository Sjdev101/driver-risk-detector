class DrowsinessAnalyzer:
    def __init__(self, ear_threshold, frame_threshold=20):
        self.ear_threshold = ear_threshold
        self.frame_threshold = frame_threshold
        self.drowsy_counter = 0
        self.is_drowsy = False
        self.prev_status = "ALERT"

    def update(self, ear):
        if ear < self.ear_threshold:
            self.drowsy_counter += 1
            if self.drowsy_counter >= self.frame_threshold:
                self.is_drowsy = True
        else:
            self.drowsy_counter = 0
            self.is_drowsy = False

        return self.is_drowsy

    def get_status(self):
        if self.is_drowsy:
            return "DROWSY"
        return "ALERT"

    def is_new_event(self):
        current_status = self.get_status()
        if current_status == "DROWSY" and self.prev_status == "ALERT":
            self.prev_status = current_status
            return True
        self.prev_status = current_status
        return False