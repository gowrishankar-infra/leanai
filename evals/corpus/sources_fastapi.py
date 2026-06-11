"""EVAL FIXTURE — source detection: FastAPI/Starlette request objects."""


async def read_item(request):
    params = request.query_params
    agent = request.headers
    body = await request.json()
    return {"params": dict(params), "ua": agent, "body": body}
