from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app import ml

description = """
The Kickstarter-Success-Predictor application deploys data science 
to indicates the liklihood of success for proposed Kickstarter campaigns.

<img src="https://miro.medium.com/max/4638/1*nOdS52xlJh2n8T2Wu0UbKg.jpeg"
width="40%" />

"""

app = FastAPI(
    title='🏆 Kickstarter-Success-Predictor',
    description=description,
    version=1.0,
    docs_url='/',
)

app.include_router(ml.router, tags=['Machine Learning'])

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['POST', 'PUT'],
#     allow_headers=['*'],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS", "DELETE", "PUT"],
    allow_headers=[
        "Access-Control-Allow-Headers",
        "Origin",
        "Accept",
        "X-Requested-With",
        "Content-Type",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods"
        "Authorization",
        "X-Amz-Date",
        "X-Api-Key",
        "X-Amz-Security-Token"
    ]
)

if __name__ == '__main__':
    uvicorn.run(app)
