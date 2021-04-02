from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app import ml

description = """
Do you have an idea that you are pretty sure will make you money? Do you believe you can get
gain support of this idea? Or are you worried your idea won't be successful to gain support?
The Kickstarter Success Predictor application deploys data science 
to indicates the liklihood of success for proposed Kickstarter campaigns. Entering the predicitive
parameters of type of idea, financial goal, and how long you need to gain backers, and our predictions,
with %92 accuracy, will tell you if you will be successful or not.
\n
<img src="https://miro.medium.com/max/4638/1*nOdS52xlJh2n8T2Wu0UbKg.jpeg"
width="40%" />
\n
\n
An app created by: Lambda Students - Frank Howd, Bryan Conn, and Azamat Jalilov - DS24
"""

app = FastAPI(
    title='🏆 Kickstarter-Success-Predictor',
    description=description,
    docs_url='/',
)

app.include_router(ml.router, tags=['Model'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
