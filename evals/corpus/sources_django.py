"""EVAL FIXTURE — source detection: Django request objects.

No sinks here. Sentinel must report ZERO findings, but its source
pass MUST register at least one input source (proves the Django
SOURCE_PATTERNS added post-M12 actually fire end-to-end).
"""


def search_view(request):
    query = request.GET.get("q", "")
    page = request.POST.get("page", "1")
    return {"query": query, "page": page}
