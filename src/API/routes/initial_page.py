from fastapi import APIRouter

initial_page = APIRouter()


@initial_page.get('/')
async def helloWorld():
    return {"Hello World!"}
