"""EVAL FIXTURE — source detection: Tornado handler arguments."""


class SearchHandler:
    def get(self):
        query = self.get_argument("q", "")
        return {"query": query}
