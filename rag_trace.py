class RAGTrace:
    def __init__(self):
        self.steps = []

    def log(self, name, data):
        self.steps.append({
            "step": name,
            "data": data
        })

    def export(self):
        return self.steps